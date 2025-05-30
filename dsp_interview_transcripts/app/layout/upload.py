import dash_bootstrap_components as dbc

from dash import dcc
from dash import html


upload_tab = html.Div(
    id="tab-upload",
    children=[
        dbc.Container(
            [
                html.Div(
                    [
                        html.H5("How to use this tab", style={"color": "#0F294A", "marginBottom": "0.5rem"}),
                        html.P("Upload the data you want to analyse.", style={"fontSize": "14px"}),
                        html.P("Supported formats:", style={"fontSize": "14px"}),
                        html.Ul(
                            [
                                html.Li(".csv or .xlsx files (you'll select which columns to use);"),
                                html.Li(".txt, .vtt or .docx files;"),
                            ],
                            style={"fontSize": "14px"},
                        ),
                    ],
                    style={"marginBottom": "2rem"},
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                    multiple=True,
                ),
                html.Div(id="upload-feedback", style={"marginTop": 10}),
                html.Hr(),
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
        ),
    ],
)
