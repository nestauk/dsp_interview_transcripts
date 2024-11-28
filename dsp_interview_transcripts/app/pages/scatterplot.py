import dash
import plotly.express as px

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

layout = html.Div(
    [
        # Scatterplot
        html.Div([dcc.Graph(id="scatter-plot")], style={"width": "100%", "marginBottom": "20px"}),
        # DataTable
        html.Div(
            [
                dash_table.DataTable(
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    id="filtered-table",
                    columns=[{"name": i, "id": i} for i in ["uuid", "role", "text"]],
                    data=[],
                    style_data_conditional=[],
                    page_action="none",
                    style_table={"height": "500px", "overflowY": "auto"},
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
    )
    # fig.update_layout(transition_duration=500)

    # Code that lets you do something if a point is clicked
    # if clickData:
    #     print(clickData)
    #     fig.add_scatter(
    #         x=[clickData["points"][0]["x"]],
    #         y=[clickData["points"][0]["y"]],
    #         mode="markers",
    #         marker=dict(size=20, color="Yellow"),
    #         name="Selected Point",
    #     )
    return fig


@dash.callback(
    [Output("filtered-table", "data"), Output("filtered-table", "style_data_conditional")],
    Input("scatter-plot", "clickData"),
)
def display_click_data(clickData):
    if clickData:
        selected_uuid = clickData["points"][0]["customdata"][2]
        conversation_id = clickData["points"][0]["customdata"][0]
        filtered_data = transcripts[transcripts["conversation"] == conversation_id]
        table_data = filtered_data[["uuid", "role", "text"]].to_dict("records")
        style_data_conditional = [
            {
                "if": {"filter_query": f'{{uuid}} = "{selected_uuid}"'},
                "backgroundColor": "#FFDDC1",
                "fontWeight": "bold",
            }
        ]
        return table_data, style_data_conditional
    return [], []
