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

from utils.dash_utils import *
from utils.pipeline import build_question_prompt_dict
from utils.pipeline import convert_transcripts_df_to_dict
from utils.pipeline import normalize_uuid
from utils.pipeline import run_batch_check
from utils.summarize import summarize_and_quote


PROMPT_PATH = Path("prompts/llm_check_system_a.txt")

dash.register_page(__name__, path="/llm_analysis", name="RQ Analysis")

layout = html.Div(
    [
        html.H3("Submit Research Questions"),
        dcc.Textarea(
            id="rq-textarea",
            placeholder="Enter RQs, one per line...",
            style={"width": "100%", "height": "150px"},
        ),
        html.Br(),
        dbc.Checkbox(id="test-mode-toggle", label="Run in test mode (no LLM calls)", value=True),
        html.Br(),
        dbc.Button("Run Analysis", id="run-analysis", color="primary"),
        html.Br(),
        html.Div(id="analysis-results"),
    ]
)


@callback(
    Output("analysis-results", "children"),
    Output("stored-output-paths", "data"),
    Output("stored-rqs", "data"),
    Input("run-analysis", "n_clicks"),
    State("stored-data", "data"),  # contents
    State("stored-column-info", "data"),
    State("rq-textarea", "value"),
    State("test-mode-toggle", "value"),  # if running in test mode, don't run the LLM
    State("session-id", "data"),
)
def run_analysis(n_clicks, contents, column_info, rq_text, test_mode, session_id):
    """
    Runs the LLM analysis (batch check + summarization and extraction of key quotes)
    whenever the "Run Analysis" button is clicked.
    """

    if not n_clicks or not contents or not column_info or not rq_text:
        return "", None, None

    df = read_data(contents)

    conv_id, role_col, text_col = column_info["conv_id"], column_info["role_col"], column_info["text_col"]
    uuid_col = column_info.get("uuid_col")

    if uuid_col == "None" or not uuid_col:
        df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
        uuid_col = "uuid"
    df[uuid_col] = df[uuid_col].apply(normalize_uuid)

    # Parse research questions
    research_questions = rq_text.strip().splitlines()
    rq_dict = {f"rq_{i+1}": q for i, q in enumerate(research_questions)}

    prompt_template = PROMPT_PATH.read_text()
    conversation_dict = convert_transcripts_df_to_dict(df, conv_id, role_col, text_col, uuid_col)
    prompt_dict = build_question_prompt_dict(rq_dict, prompt_template)

    # === TEST MODE: skip LLM ===
    if test_mode:
        # Try loading cached files if they exist
        output_paths = {}
        for rq_id in rq_dict:
            mock_path = Path("mock_outputs") / "rq_1_output.jsonl"
            if mock_path.exists():
                output_paths[rq_id] = str(mock_path)
            else:
                # Fallback mock data

                # grab some random quotes from the original data
                all_quotes = df[text_col].to_list()
                mock_quotes = random.sample(all_quotes, 3)

                mock_path.parent.mkdir(parents=True, exist_ok=True)
                with open(mock_path, "w") as f:
                    json.dump(
                        {
                            "rq_1": "yes",
                            "explanation": "This is a test explanation",
                            "text": mock_quotes,
                            "identifier": str(uuid.uuid4()) * len(mock_quotes),
                            "id": str(uuid.uuid4()),
                            "timestamp": "2025-04-07T00:00:00Z",
                            "model": "mock",
                            "temperature": 0,
                        },
                        f,
                    )
                output_paths[rq_id] = str(mock_path)

        return dbc.Alert("Test mode: using mock outputs", color="info"), output_paths, rq_dict
    else:
        # === NORMAL MODE: run LLM ===
        output_dir = os.path.join("outputs", session_id)
        os.makedirs(output_dir, exist_ok=True)
        output_paths = run_batch_check(conversation_dict, prompt_dict, output_dir)
        return dbc.Alert("LLM processing complete! See below for results.", color="success"), output_paths, rq_dict


@callback(
    Output("analysis-results", "children", allow_duplicate=True),
    Input("stored-output-paths", "data"),
    State("stored-rqs", "data"),
    State("test-mode-toggle", "value"),  # if running in test mode, don't run the LLM
    prevent_initial_call="initial_duplicate",
)
def display_results(output_paths, rq_dict, test_mode):
    """Displays the summary answer and extracted quotes for each RQ."""
    if not output_paths or not rq_dict:
        return ""

    children = []

    for rq_id, question in rq_dict.items():
        path = Path(output_paths[rq_id])
        if not path.exists():
            children.append(html.Div(f"No output found for: {question}", style={"color": "red"}))
            continue

        df = pd.read_json(path, lines=True)

        extracted_texts = [txt for sublist in df["text"] for txt in sublist]

        if test_mode:
            answer = f"(TEST) This is a mock summary for: {question}"
            quotes = extracted_texts[:3]
        else:
            answer, quotes = summarize_and_quote(extracted_texts, question)

        quote_elements = []
        for i, row in df.iterrows():
            for j, (quote, identifier) in enumerate(zip(row["text"], row["identifier"])):
                if any(summary_quote in quote or quote in summary_quote for summary_quote in quotes):
                    quote_elements.append(
                        html.Li(
                            quote,
                            style={"cursor": "pointer", "color": "blue", "textDecoration": "underline"},
                            id={"type": "quote", "index": f"{rq_id}::{i}::{j}"},
                        )
                    )

        children.append(html.H5(f"RQ: {question}"))
        children.append(html.P(f"**Summary Answer:** {answer}"))
        children.append(html.Ul(quote_elements))

    return (html.Div(children),)


@callback(
    Output("quote-modal", "is_open"),
    Output("modal-body", "children"),
    Input({"type": "quote", "index": ALL}, "n_clicks"),
    State("stored-output-paths", "data"),
    State("stored-rqs", "data"),
    State("stored-data", "data"),  # contents
    State("stored-column-info", "data"),
)
def display_conversation(n_clicks_list, output_paths, rq_dict, contents, column_info):
    """
    If the user clicks one of the quotes, this brings up a pop-up showing the full conversation
    with the clicked quote highlighted in yellow.
    """
    if not any(n_clicks_list):
        raise PreventUpdate

    # Identify which quote was clicked
    triggered_id = ctx.triggered_id
    if not triggered_id:
        raise PreventUpdate

    try:
        rq_id, i, j = triggered_id["index"].split("::")
        i = int(i)
        j = int(j)
    except Exception as e:
        return False, f"Error parsing index: {e}"

    path = Path(output_paths[rq_id])
    if not path.exists():
        return False, "Output not found."

    # Load original data + output
    df_output = pd.read_json(path, lines=True)
    df_original = read_data(contents)

    # Get quote and conversation ID

    quote_text = df_output.iloc[i]["text"][j]

    conv_id_col = column_info["conv_id"]
    text_col = column_info["text_col"]

    # Find the row in the original df that contains this quote
    matching_row = df_original[df_original[text_col].str.contains(quote_text, na=False)]
    if matching_row.empty:
        return True, f"Could not find the quote in the original data."

    conv_id = matching_row[conv_id_col].iloc[0]
    filtered = df_original[df_original[conv_id_col] == conv_id]

    def highlight_text(row):
        text = row[text_col]
        if quote_text in text:
            return html.Mark(text)
        return text

    conversation_display = [
        html.Div([html.Strong(f"{i+1}. "), html.Span(highlight_text(row))]) for i, row in filtered.iterrows()
    ]

    return True, html.Div(conversation_display)
