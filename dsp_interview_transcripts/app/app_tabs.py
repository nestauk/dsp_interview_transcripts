import base64
import io
import os
import re
import uuid

from collections import Counter

import dash
import dash_bootstrap_components as dbc

from callbacks.rq_callbacks import register_rq_callbacks
from callbacks.topic_modelling_callbacks import register_topic_callbacks
from callbacks.upload_callbacks import register_upload_callbacks
from dash import Input
from dash import Output
from dash import State
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from layout.rq_tab import quotes_modal
from layout.rq_tab import rq_tab
from layout.topic_mapping import topic_tab

# tab layouts
from layout.upload import upload_tab


# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Multi-Tab Interview Analysis"

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Interview Analysis", className="ms-2"),
        ]
    ),
    color="#0F294A",
    dark=True,
    className="mb-4",
)

# App layout with two tabs: Upload & Word Count
app.layout = html.Div(
    # style={"backgroundColor": "#0F294A", "minHeight": "100vh", "color": "white"},
    children=[
        # Stores to hold the uploaded DataFrame, selected columns, and session
        dcc.Store(id="data-store"),  # stores raw uploaded data
        dcc.Store(id="column-store"),  # stores {'conv_id', 'role', 'text', 'uuid'}
        dcc.Store(id="session-id"),  # session identifier
        dcc.Store(id="stored-topic-viz"),  # dataframe for scatter plot
        dcc.Store(id="stored-output-paths"),  # for RQ analysis
        dcc.Store(id="stored-rqs"),  # for RQ analysis
        dcc.Store(id="output-dir"),  # for RQ analysis
        dbc.Container(
            [
                navbar,
                # Tabs component
                dcc.Tabs(
                    id="tabs",
                    value="tab-upload",
                    children=[
                        dcc.Tab(label="1. Upload data ⬆️", value="tab-upload", children=[]),
                        dcc.Tab(label="2. Deductive / topic mapping analysis 🕸️", value="tab-topic", children=[]),
                        dcc.Tab(label="3. Inductive / framework analysis 🔍", value="tab-rq", children=[]),
                    ],
                ),
                # html.Div(id="tab-content"),
                html.Div(
                    [
                        upload_tab,
                        topic_tab,
                        rq_tab,
                    ],
                ),
                # Modal for quotes
                quotes_modal,
            ],
            fluid=True,
            style={"maxWidth": "1200px"},
        ),
    ]
)


# Callback to switch visible tab
@app.callback(
    Output("tab-upload", "style"), Output("tab-topic", "style"), Output("tab-rq", "style"), Input("tabs", "value")
)
def switch_tab(tab):
    return (
        {"display": "block"} if tab == "tab-upload" else {"display": "none"},
        {"display": "block"} if tab == "tab-topic" else {"display": "none"},
        {"display": "block"} if tab == "tab-rq" else {"display": "none"},
    )


register_upload_callbacks(app)
register_topic_callbacks(app)
register_rq_callbacks(app)


if __name__ == "__main__":
    app.run(debug=True)
