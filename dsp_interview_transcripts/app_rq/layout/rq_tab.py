import dash_bootstrap_components as dbc

from dash import dcc
from dash import html


rq_tab = html.Div(
    id="tab-rq",
    children=[
        html.H3("Research Question Analysis"),
        dcc.Textarea(
            id="rq-textarea",
            placeholder="Enter RQs, one per line...",
            style={"width": "100%", "height": "150px", "marginBottom": "1rem"},
        ),
        dbc.Checkbox(id="test-mode-toggle", label="Run in test mode (no LLM calls)", value=True),
        html.Br(),
        dbc.Button("Run Analysis", id="run-analysis", className="mt-2 nesta-button"),
        html.Div(id="download-btn-container"),
        html.Br(),
        dbc.Spinner(html.Div(id="analysis-results"), size="md"),
        dcc.Download(id="download-results"),
    ],
    className="mt-4",
    style={"display": "none"},
)

quotes_modal = dbc.Modal(
    [dbc.ModalHeader(dbc.ModalTitle("Conversation View")), dbc.ModalBody(id="modal-body")],
    id="quote-modal",
    size="xl",
    is_open=False,
)
