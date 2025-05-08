import dash_bootstrap_components as dbc

from dash import dash_table
from dash import dcc
from dash import html


topic_tab = html.Div(
    id="tab-topic",
    children=[
        html.H4("Topic Mapping"),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Number of topics"),
                                dcc.Input(
                                    id="num-topics-input",
                                    type="number",
                                    min=2,
                                    max=100,
                                    step=1,
                                    value=10,
                                    style={"width": "100%"},
                                ),
                            ],
                            width=3,
                        ),
                        dbc.Col(dbc.Button("Run", id="run-topic-model-btn", className="mt-2 nesta-button"), width=2),
                        dbc.Col(dbc.Spinner(html.Div(id="topic-model-status"), size="sm", color="info"), width=7),
                    ]
                )
            ),
            className="mb-3",
        ),
        html.Div(
            id="topic-results",
            style={"display": "none"},  # by default, the results are hidden. They are revealed once the model has run.
            children=[
                html.Div(
                    [
                        html.H4(
                            "Topic descriptions and key words",
                            style={"color": "#0F294A", "fontWeight": "bold", "marginTop": "20px"},
                        ),
                        dash_table.DataTable(
                            id="topic-lookup-table",
                            page_action="none",
                            style_table={"height": "500px", "overflowY": "auto", "overflowX": "auto"},
                            style_cell={
                                "fontFamily": "Century Gothic",
                                "fontSize": "14px",
                                "textAlign": "left",
                            },
                            style_header={
                                "backgroundColor": "#0F294A",
                                "color": "white",
                                "fontWeight": "bold",
                                "textAlign": "center",
                            },
                            style_data={
                                "whiteSpace": "normal",
                                "height": "auto",
                                "fontFamily": "Century Gothic",
                                "fontSize": "14px",
                                "color": "#0F294A",
                            },
                            sort_action="native",
                        ),
                        html.Br(),
                        dbc.Button(
                            "Download Topics as CSV",
                            id="download-topic-csv-btn",
                            className="nesta-button",
                        ),
                        dcc.Download(id="download-topic-csv"),
                    ]
                ),
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
                                "padding": "20px",
                                "borderRight": "2px solid #ccc",
                                "backgroundColor": "#F6F8FA",
                                "fontFamily": "Century Gothic",
                                "fontSize": "14px",
                                "color": "#0F294A",
                            },
                            children=[
                                html.H4("Selected Point Info", style={"color": "#0F294A", "fontWeight": "bold"}),
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
                        html.P(
                            "When you click a point on the plot, you will see the full text of that conversation below."
                        ),
                        html.Div(id="conversation-view", style={"marginTop": "20px"}),
                    ]
                ),
            ],
        ),
    ],
    className="mt-4",
    style={"display": "none"},
)
