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
from style import CONTENT_STYLE
from style import NESTA_COLOURS
from style import SIDEBAR_STYLE


dash.register_page(__name__, path="/scatterplot")


layout = html.Div(
    [
        html.Div(
            children=[
                html.P(
                    "This tab contains an interactive visualisation to help you explore user responses within each topic. Each user response is shown as a point."
                ),
                html.P(
                    "Click a point on the plot to find out more information about it. On the left, you will see information about the topic it is in, "
                    "the ID of the conversation it occurred in, and the response itself."
                ),
            ]
        ),
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
                        html.Div(id="conversation-display", style={"marginBottom": "10px"}),
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
        html.Div(
            children=[
                html.P("When you click a point on the plot, this table will show the full text of that conversation."),
            ]
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

    # Update hovertemplate to show only 'text_clean'
    fig.update_traces(hovertemplate="<b>%{customdata[1]}</b><extra></extra>")

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
        legend_title_text="",  # Hide legend title
    )

    return fig


@dash.callback(
    [
        Output("filtered-table", "data"),
        Output("filtered-table", "style_data_conditional"),
    ],
    Input("scatter-plot", "clickData"),
)
def update_table(clickData):
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

        return table_data, style_data_conditional

    return [], []


@dash.callback(
    [
        Output("name-display", "children"),
        Output("description-display", "children"),
        Output("conversation-display", "children"),
        Output("text-clean-display", "children"),
    ],
    Input("scatter-plot", "clickData"),
)
def update_point_info(clickData):
    if clickData:
        selected_uuid = clickData["points"][0]["customdata"][2]
        conversation_id = clickData["points"][0]["customdata"][0]

        selected_point = data_viz[data_viz["uuid"] == selected_uuid].iloc[0]
        name = f"Topic name: {selected_point['Name']}"
        description = f"Topic description: {selected_point.get('Description', 'N/A')}"
        conversation = f"Conversation ID: {conversation_id}"
        text_clean = f"User response: {selected_point['text_clean']}"

        return name, description, conversation, text_clean

    return "Topic name: N/A", "Topic description: N/A", "Conversation ID: N/A", "User response: N/A"
