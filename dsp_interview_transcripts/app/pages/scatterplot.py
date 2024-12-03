import dash
import plotly.express as px
import plotly.graph_objects as go

from dash import Input
from dash import Output
from dash import dash_table
from dash import dcc
from dash import html
from data import data_viz
from data import transcripts


dash.register_page(__name__, path="/scatterplot")

# Sidebar layout
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

NESTA_COLOURS = [
    "#0000FF",
    "#FDB633",
    "#18A48C",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#FFFFFF",
    "#000000",
]

layout = html.Div(
    [
        # Scatterplot
        # html.Div([dcc.Graph(id="scatter-plot")], style={"width": "100%", "marginBottom": "20px"}),
        # First row: Information Panel and Scatterplot
        html.Div(
            [
                # Information panel (left one-third)
                html.Div(
                    id="info-panel",
                    style={
                        "width": "25%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "10px",
                        "borderRight": "1px solid #ccc",
                        "backgroundColor": "#f9f9f9",
                    },
                    children=[
                        html.H4("Selected Point Info"),
                        html.Div(id="name-display", style={"marginBottom": "10px"}),
                        html.Div(id="description-display", style={"marginBottom": "10px"}),
                        html.Div(id="text-clean-display", style={"marginBottom": "10px"}),
                    ],
                ),
                # Scatterplot (right two-thirds)
                html.Div(
                    [dcc.Graph(id="scatter-plot")],
                    style={"width": "75%", "display": "inline-block"},
                ),
            ],
            style={"width": "100%", "marginBottom": "20px"},
        ),
        # DataTable
        html.Div(
            [
                dash_table.DataTable(
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    id="filtered-table",
                    columns=[{"name": i, "id": i} for i in ["uuid", "role", "text_clean"]],
                    data=[],
                    style_data_conditional=[],
                    page_action="none",
                    style_table={"height": "500px", "overflowY": "auto"},
                    style_cell={
                        "fontFamily": "Century Gothic",  # Set font to Century Gothic
                        "fontSize": "14px",  # Optional: set font size
                        "textAlign": "left",  # Optional: align text
                    },
                )
            ],
            style={"width": "100%"},
        ),
    ],
    style=CONTENT_STYLE,
)


@dash.callback(Output("scatter-plot", "figure"), [Input("scatter-plot", "clickData")], prevent_initial_call=False)
def update_scatter_plot(clickData):

    fig = px.scatter(
        data_viz,
        x="x",
        y="y",
        color="Name",
        hover_data=["conversation", "text_clean"],
        custom_data=["conversation", "text_clean", "uuid"],
        color_discrete_sequence=NESTA_COLOURS,
    )

    # Update individual traces for opacity - for some reason you can't set an opacity variable for the whole plot
    for trace in fig.data:
        # Match points in each trace and set opacity
        trace_name = trace.name  # Get the category name for this trace
        trace_opacity = data_viz[data_viz["Name"] == trace_name]["opacity"].values
        trace.update(marker=dict(opacity=trace_opacity))

    # Code to highlight the point that's been clicked
    if clickData:

        fig.add_trace(
            go.Scatter(
                x=[clickData["points"][0]["x"]],
                y=[clickData["points"][0]["y"]],
                mode="markers",
                marker=dict(size=20, color="Yellow"),
                showlegend=False,
            ),
        )

    fig.update_layout(
        uirevision="scatter-plot",  # Ensure that the zoom level is preserved after you've clicked a point
        xaxis=dict(showticklabels=False, title_text=""),  # Hide x-axis ticks and title
        yaxis=dict(showticklabels=False, title_text=""),  # Hide y-axis ticks and title
    )

    return fig


@dash.callback(
    [
        Output("filtered-table", "data"),
        Output("filtered-table", "style_data_conditional"),
        Output("name-display", "children"),
        Output("description-display", "children"),
        Output("text-clean-display", "children"),
    ],
    Input("scatter-plot", "clickData"),
)
def display_click_data(clickData):
    if clickData:
        selected_uuid = clickData["points"][0]["customdata"][2]
        conversation_id = clickData["points"][0]["customdata"][0]

        filtered_data = transcripts[transcripts["conversation"] == conversation_id]
        table_data = filtered_data[["uuid", "role", "text_clean"]].to_dict("records")
        style_data_conditional = [
            {
                "if": {"filter_query": f'{{uuid}} = "{selected_uuid}"'},
                "backgroundColor": "#FFFF00",
                "fontWeight": "bold",
            }
        ]

        selected_point = data_viz[data_viz["uuid"] == selected_uuid].iloc[0]
        name = f"Name: {selected_point['Name']}"
        description = f"Description: {selected_point.get('Description', 'N/A')}"
        text_clean = f"Text: {selected_point['text_clean']}"

        return table_data, style_data_conditional, name, description, text_clean

    return [], [], "Name: N/A", "Description: N/A", "Text: N/A"
