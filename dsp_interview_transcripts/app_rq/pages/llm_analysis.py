import json
import os
import random
import uuid

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd

from dash import Input
from dash import Output
from dash import State
from dash import callback
from dash import ctx
from dash import dcc
from dash import html
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
from style import CONTENT_STYLE
from style import NESTA_COLOURS
from style import SIDEBAR_STYLE

from dsp_interview_transcripts import PROJECT_DIR
from utils.dash_utils import get_cleaned_data
from utils.dash_utils import get_or_create_output_dir
from utils.llm_question_answering import concat_batch_check_output
from utils.llm_question_answering import normalize_uuid
from utils.llm_question_answering import run_batch_check_for_all_rqs
from utils.llm_summarize import create_output_excel
from utils.llm_summarize import generate_full_summary_output
from utils.llm_summarize import generate_summaries


dash.register_page(__name__, path="/llm_analysis", name="RQ Analysis")

layout = html.Div(
    [
        html.Div(
            [
                html.H5("How to use this tool", style={"color": "#0F294A", "marginBottom": "0.5rem"}),
                html.P(
                    "This tool allows you to submit your own research questions (RQs) and receive AI-generated summaries and illustrative quotes from your interview data.",
                    style={"fontSize": "14px", "marginBottom": "0.5rem"},
                ),
                html.Ul(
                    [
                        html.Li("Enter one or more research questions in the box below (one question per line)."),
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
            style={"marginBottom": "1.5rem"},
        ),
        html.H4("Submit Research Questions", style={"color": "#0F294A", "fontWeight": "bold"}),
        dcc.Textarea(
            id="rq-textarea",
            placeholder="Enter RQs, one per line...",
            style={
                "width": "100%",
                "height": "150px",
                "border": "1px solid #ccc",
                "padding": "10px",
                "fontFamily": "Century Gothic",
                "fontSize": "14px",
            },
        ),
        html.Br(),
        dbc.Checkbox(id="test-mode-toggle", label="Run in test mode (no LLM calls)", value=True),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Button("Run Analysis", id="run-analysis", className="nesta-button")),
                dbc.Col(html.Div(id="download-btn-container")),
            ]
        ),
        html.Br(),
        dbc.Spinner(html.Div(id="analysis-results"), size="md", color="primary", type="border"),
        html.Br(),
        dcc.Download(id="download-results"),
    ],
    style={
        **CONTENT_STYLE,
        "width": "85%",
        "margin": "0",
        "padding": "2rem",
        "fontFamily": "Century Gothic",
        "color": "#0F294A",
    },
)


@callback(
    Output("analysis-results", "children"),
    Output("stored-output-paths", "data"),
    Output("stored-rqs", "data"),
    Output("output-dir", "data"),
    Output("download-btn-container", "children"),
    Input("run-analysis", "n_clicks"),
    State("session-id", "data"),
    State("stored-column-info", "data"),
    State("rq-textarea", "value"),
    State("test-mode-toggle", "value"),  # if running in test mode, don't run the LLM
    prevent_initial_call=True,
)
def run_analysis(n_clicks, session_id, column_info, rq_text, test_mode):
    """
    Runs the LLM analysis (batch check + summarization and extraction of key quotes)
    whenever the "Run Analysis" button is clicked.

    TODO: move all of this to FastAPI
    """

    if not n_clicks or not session_id or not column_info or not rq_text:
        return "", None, None, None, None

    df = get_cleaned_data(session_id)

    conv_id, role_col = column_info["conv_id"], column_info["role_col"]
    uuid_col = column_info.get("uuid_col")

    if uuid_col == "None" or not uuid_col:
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        uuid_col = "uuid"
    df[uuid_col] = df[uuid_col].apply(normalize_uuid)

    output_dir = get_or_create_output_dir(session_id, test_mode=test_mode)

    # Step 1: batch_check
    output_paths, rq_dict = run_batch_check_for_all_rqs(
        rq_text=rq_text,
        cleaned_df=df,
        output_dir=output_dir,
        conv_col="conversation",
        role_col="role",
        uuid_col="uuid",
    )

    # TODO: make this available for download
    df_output = concat_batch_check_output(rq_dict, output_paths)

    # Generate summaries ==============================
    per_rq_outputs, long_dfs = generate_summaries(
        rq_dict, output_paths, output_dir, test_mode, "text"
    )  # doesn't call an LLM if we're in test mode

    # Generate a table with answers and quotes for all RQs
    full_summary_df = generate_full_summary_output(rq_dict, long_dfs, per_rq_outputs, "text")
    full_summary_df.to_csv(f"{output_dir}/full_summary.csv", index=False)

    # TODO: create a warning if this isn't true
    print(f"All quotes found in all returned texts: {(full_summary_df['text'] == full_summary_df['quotes']).all()}")

    # Save the output as excel
    create_output_excel(full_summary_df, output_dir)

    download_btn = dbc.Button("Download Results", id="trigger-download", className="nesta-button", n_clicks=0)

    if test_mode:
        return (
            dbc.Alert("Test mode: using mock outputs", color="info"),
            output_paths,
            rq_dict,
            output_dir,
            download_btn,
        )
    else:
        return (
            dbc.Alert("LLM processing complete! See below for results.", color="success"),
            output_paths,
            rq_dict,
            output_dir,
            download_btn,
        )


# handle the file download
@callback(
    Output("download-results", "data"),
    Input("trigger-download", "n_clicks"),
    State("output-dir", "data"),
    prevent_initial_call=True,
)
def download_excel(n_clicks, output_dir):
    """
    Download the excel table of summaries for each RQ when the download button is cicked
    """
    if not output_dir or not n_clicks or n_clicks < 1:
        raise PreventUpdate
    return dcc.send_file(f"{output_dir}/full_summary.xlsx")


@callback(
    Output("analysis-results", "children", allow_duplicate=True),
    Input("output-dir", "data"),
    State("stored-rqs", "data"),
    State("session-id", "data"),
    State("stored-column-info", "data"),
    prevent_initial_call="initial_duplicate",
)
def display_results(output_dir, rq_dict, session_id, column_info):
    """
    Displays the summary answer and extracted quotes for each RQ.
    """
    if not output_dir or not rq_dict:
        return ""

    children = []

    full_summary_df = pd.read_csv(f"{output_dir}/full_summary.csv")
    df_original = get_cleaned_data(session_id)

    uuid_col = column_info.get("uuid_col", "uuid")
    if uuid_col == "None":
        uuid_col = "uuid"

    valid_ids = set(df_original[uuid_col].dropna().unique())

    for _, question in rq_dict.items():

        temp_df = full_summary_df[full_summary_df["question"] == question]

        if len(temp_df) == 0:
            children.append(html.Div(f"No output found for: {question}", style={"color": "red"}))
            continue

        # Filter out unmatched or blank quotes
        temp_df = temp_df[
            temp_df["identifier"].isin(valid_ids)
            & temp_df["text"].notnull()
            & temp_df["text"].apply(
                lambda x: isinstance(x, str) and x.strip() != ""
            )  # (temp_df["text"].str.strip() != "")
        ]

        if temp_df.empty:
            children.append(html.Div(f"No valid quotes for: {question}", style={"color": "orange"}))
            continue

        quote_elements = [
            html.Div(
                f"{row['text']}",
                id={"type": "quote", "index": row["identifier"]},
                className="quote-block",
            )
            for _, row in temp_df.iterrows()
        ]

        children.append(html.H4(f"RQ: {question}", style={"color": "#0F294A", "marginTop": "2rem"}))

        # Add small heading for the summary
        children.append(html.H6("Summary (LLM-generated)", style={"color": "#646363", "marginBottom": "0.25rem"}))

        children.append(html.P(temp_df["answer"].values[0], style={"marginBottom": "1rem", "color": "#0F294A"}))

        # Add small heading for quotes
        children.append(html.H6("Illustrative quotes", style={"color": "#646363", "marginBottom": "0.25rem"}))

        children.append(html.Div(quote_elements))

    return (html.Div(children),)


@callback(
    Output("quote-modal", "is_open"),
    Output("modal-body", "children"),
    Input({"type": "quote", "index": ALL}, "n_clicks"),
    State("session-id", "data"),
    State("stored-column-info", "data"),
)
def display_conversation(n_clicks_list, session_id, column_info):
    """
    If the user clicks one of the quotes, this brings up a pop-up showing the full conversation
    with the clicked quote highlighted in yellow.
    """
    if not any(n_clicks_list):
        raise PreventUpdate

    # Identify which quote was clicked
    triggered = ctx.triggered_id
    if not triggered or "index" not in triggered:
        raise PreventUpdate

    uuid_clicked = triggered["index"]

    df_original = get_cleaned_data(session_id)

    conv_id_col = column_info["conv_id"]
    role_col = column_info["role_col"]
    uuid_col = column_info.get("uuid_col", "uuid")
    if uuid_col == "None":
        uuid_col = "uuid"

    # Find the row in the original df that contains this quote
    matching_row = df_original[df_original[uuid_col] == uuid_clicked]
    if matching_row.empty:
        return True, f"Could not find the quote in the original data."

    conv_id = matching_row[conv_id_col].iloc[0]
    quote_text = matching_row["text_clean"].iloc[0]

    convo_df = df_original[df_original[conv_id_col] == conv_id]

    def highlight_text(row):
        text = row["text_clean"]
        if quote_text in text:
            return html.Mark(text)
        return text

    conversation_display = [
        html.Div([html.Strong(f"{row[role_col]}. "), html.Span(highlight_text(row))], style={"marginBottom": "0.5rem"})
        for _, row in convo_df.iterrows()
    ]

    return True, html.Div(conversation_display)
