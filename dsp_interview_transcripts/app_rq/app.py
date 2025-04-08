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
        html.H2("Interview analysis", className="display-5"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("Upload & Select Columns", href="/", active="exact"),
                dbc.NavLink("Topic mapping", href="/topic_modelling", active="exact"),
                dbc.NavLink("RQ Analysis", href="/llm_analysis", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Main layout
app.layout = html.Div(
    [
        dcc.Location(id="url"),  # Not strictly required for Dash pages but good practice
        sidebar,
        html.Div(
            page_container,
            style=CONTENT_STYLE,
        ),
        # Global storage and modal (outside page_container to persist across pages)
        dcc.Store(id="stored-data"),  # contains "contents" i.e. encoded content of uploaded csv
        dcc.Store(id="stored-column-info"),  # names for the columns in the input data
        dcc.Store(id="stored-output-paths"),  # LLM analysis: one output path per RQ
        dcc.Store(id="stored-rqs"),
        dcc.Store(id="stored-original-df"),
        dcc.Store(id="session-id"),  # session ID for each user
        dcc.Store(id="stored-topic-viz"),  # output df_vis from topic modelling
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
)

# app.layout = dbc.Container(
#     [dbc.Row([dbc.Col(sidebar, width=3), dbc.Col(page_container, width=9)])],
#     fluid=True,
# )

# app.layout.children += [
#     dcc.Store(id="stored-data"),  # contains "contents" i.e. encoded content of uploaded csv
#     dcc.Store(id="stored-column-info"),  # names for the columns in the data
#     dcc.Store(id="stored-output-paths"),  # one output path per RQ
#     dcc.Store(id="stored-rqs"),
#     dcc.Store(id="stored-original-df"),
#     dcc.Store(id="session-id"),  # session ID for each user
#     # store the clicked quote
#     dbc.Modal(
#         [
#             dbc.ModalHeader(dbc.ModalTitle("Conversation View")),
#             dbc.ModalBody(id="modal-body"),
#         ],
#         id="quote-modal",
#         size="xl",
#         is_open=False,
#     ),
# ]

if __name__ == "__main__":
    app.run(debug=True)
