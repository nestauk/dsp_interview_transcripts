import dash
import dash_bootstrap_components as dbc

from dash import dcc
from dash import html
from dash import page_container
from style import CONTENT_STYLE
from style import NESTA_COLOURS
from style import SIDEBAR_STYLE


# from pathlib import Path
# import os
# import uuid

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Multi-page Interview Analysis App"

sidebar = html.Div(
    [
        dbc.Nav(
            [
                dbc.NavLink("Upload & Select Columns", href="/", active="exact"),
                dbc.NavLink("Topic mapping", href="/topic_modelling", active="exact"),
                dbc.NavLink("RQ Analysis", href="/llm_analysis", active="exact"),
            ],
            vertical=True,
            pills=True,
        )
    ],
    style=SIDEBAR_STYLE,
)

app.layout = dbc.Container(
    [dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(page_container, width=9)])],
    fluid=True,
)

app.layout.children += [
    dcc.Store(id="stored-data"),  # contains "contents" i.e. encoded content of uploaded csv
    dcc.Store(id="stored-column-info"),  # names for the columns in the data
    dcc.Store(id="stored-output-paths"),  # one output path per RQ
    dcc.Store(id="stored-rqs"),
    dcc.Store(id="stored-original-df"),
    dcc.Store(id="session-id"),  # session ID for each user
    # store the clicked quote
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Conversation View")),
            dbc.ModalBody(id="modal-body"),
        ],
        id="quote-modal",
        size="xl",
        is_open=False,
    ),
]

if __name__ == "__main__":
    app.run(debug=True)
