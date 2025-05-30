# === app.py ===
import base64
import io
import os
import uuid

import dash
import dash_auth
import dash_bootstrap_components as dbc

from callbacks.rq_callbacks import register_rq_callbacks
from callbacks.topic_modelling_callbacks import register_topic_callbacks
from callbacks.upload_callbacks import register_upload_callbacks
from dash import Input
from dash import Output
from dash import dcc
from dash import html
from dotenv import load_dotenv
from layout.rq_tab import quotes_modal
from layout.rq_tab import rq_tab
from layout.topic_mapping import topic_tab
from layout.upload import upload_tab


load_dotenv()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(app, {os.environ.get("VALID_USERNAME"): os.environ.get("VALID_PASSWORD")})

navbar = dbc.Navbar(
    dbc.Container([dbc.NavbarBrand("Interview Analysis", className="ms-2")]),
    color="#0F294A",
    dark=True,
    className="mb-4",
)

app.layout = html.Div(
    [
        dcc.Store(id="data-store"),
        dcc.Store(id="column-store"),
        dcc.Store(id="session-id"),
        dcc.Store(id="stored-topic-viz"),
        dcc.Store(id="stored-output-paths"),
        dcc.Store(id="stored-rqs"),
        dcc.Store(id="output-dir"),
        dcc.Store(id="show-column-select-flag"),
        dbc.Container(
            [
                navbar,
                dcc.Tabs(
                    id="tabs",
                    value="tab-upload",
                    children=[
                        dcc.Tab(label="1. Upload data ⬆️", value="tab-upload"),
                        dcc.Tab(label="2. Deductive / topic mapping analysis 🔸", value="tab-topic"),
                        dcc.Tab(label="3. Inductive / framework analysis 🔍", value="tab-rq"),
                    ],
                ),
                html.Div([upload_tab, topic_tab, rq_tab]),
                quotes_modal,
            ],
            fluid=True,
            style={"maxWidth": "1200px"},
        ),
    ]
)


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
    app.run(host="0.0.0.0", debug=True)
