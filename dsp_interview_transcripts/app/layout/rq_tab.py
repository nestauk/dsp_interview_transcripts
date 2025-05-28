import dash_bootstrap_components as dbc

from dash import dcc
from dash import html


rq_tab = html.Div(
    id="tab-rq",
    children=[
        dbc.Container(
            [
                html.Div(
                    [
                        html.H5("How to use this tab", style={"color": "#0F294A", "marginBottom": "0.5rem"}),
                        html.P(
                            "This tab allows you to submit your own research questions and receive AI-generated summaries and illustrative quotes from your interview data.",
                            style={"fontSize": "14px", "marginBottom": "0.5rem"},
                        ),
                        html.P(
                            "Data is processed using Azure OpenAI (i.e. Microsoft's version of GPT). This is more secure than using OpenAI directly.",
                            style={"fontSize": "14px", "marginBottom": "0.5rem"},
                        ),
                        html.P(
                            "To use this tab:",
                            style={"fontSize": "14px", "marginBottom": "0.5rem"},
                        ),
                        html.Ul(
                            [
                                html.Li(
                                    "Enter one or more research questions in the box below (one question per line)."
                                ),
                                html.Li(
                                    "Click 'Run Analysis' to generate summaries and pull out relevant quotes from the dataset."
                                ),
                                html.Li("Click on any quote to view the full conversation it came from."),
                                html.Li(
                                    "When you're happy with the results, click 'Download Results' to export a summary table."
                                ),
                            ],
                            style={"fontSize": "14px", "marginBottom": "1.5rem"},
                        ),
                    ],
                    style={"marginBottom": "2rem"},
                ),
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
            className="nesta-tab-container",
            fluid=True,
        ),
    ],
    style={"display": "none"},
)

quotes_modal = dbc.Modal(
    [dbc.ModalHeader(dbc.ModalTitle("Conversation View")), dbc.ModalBody(id="modal-body")],
    id="quote-modal",
    size="xl",
    is_open=False,
)
