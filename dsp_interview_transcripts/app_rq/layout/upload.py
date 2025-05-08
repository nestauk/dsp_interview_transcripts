import dash_bootstrap_components as dbc

from dash import dcc
from dash import html


upload_tab = html.Div(
    id="tab-upload",
    children=[
        html.H3("Upload CSV"),
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
    className="mt-4",
    style={"display": "block"},
)
