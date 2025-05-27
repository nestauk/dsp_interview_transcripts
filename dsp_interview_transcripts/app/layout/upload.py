import dash_bootstrap_components as dbc

from dash import dcc
from dash import html


upload_tab = html.Div(
    id="tab-upload",
    children=[
        dbc.Container(
            [
                # html.H3("Upload CSV"),
                html.Div(
                    [
                        html.H5("How to use this tab", style={"color": "#0F294A", "marginBottom": "0.5rem"}),
                        html.P(
                            "Upload the data you want to analyse. ",
                            style={"fontSize": "14px", "marginBottom": "0.5rem"},
                        ),
                        html.P(
                            "The data must meet these requirements:",
                            style={"fontSize": "14px", "marginBottom": "0.5rem"},
                        ),
                        html.Ul(
                            [
                                html.Li("The data should be **tabular**, e.g. .csv format;"),
                                html.Li("The data should contain a column that identifies the user/conversation;"),
                                html.Li(
                                    "The data should contain a column that identifies the role of each speaker, i.e. interviewer or participant;"
                                ),
                                html.Li(
                                    "The data should contain a column that contains the text of the conversation."
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "1rem"},
                        ),
                        html.P(
                            "Optionally, there can also be a column that uniquely identifies each response.",
                            style={"fontSize": "14px", "marginBottom": "0.5rem"},
                        ),
                    ],
                    style={"marginBottom": "2rem"},
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select a CSV File")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                    multiple=False,
                ),
                html.Div(id="upload-feedback", style={"marginTop": 10}),
                html.Hr(),
                # Column selectors
                html.Div(
                    [
                        html.H5("Select columns"),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Dropdown(id="conv-id-dropdown", placeholder="Conversation ID column")),
                                dbc.Col(dcc.Dropdown(id="role-dropdown", placeholder="Role column")),
                                dbc.Col(dcc.Dropdown(id="text-dropdown", placeholder="Text column")),
                                dbc.Col(dcc.Dropdown(id="uuid-dropdown", placeholder="UUID column")),
                            ]
                        ),
                        html.Br(),
                        dbc.Button("Save", id="save-btn", color="primary"),
                        html.Div(id="save-feedback", style={"marginTop": 10, "color": "green"}),
                    ],
                    id="column-section",
                    style={"display": "none"},
                ),
            ],
            className="nesta-tab-container",
            fluid=True,
        )
    ],
)
