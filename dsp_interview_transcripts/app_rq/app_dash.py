import json
import os
import uuid

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd

from dash import Input
from dash import Output
from dash import State
from dash import ctx
from dash import dash_table
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

user_id = str(uuid.uuid4())
session_output_dir = os.path.join("outputs", user_id)
os.makedirs(session_output_dir, exist_ok=True)

OUTPUT_DIR = session_output_dir

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Framework / top-down analysis"

app.layout = dbc.Container(
    [
        html.H2("Research Question Explorer"),
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
        html.Div(id="column-selectors"),
        html.Br(),
        dcc.Textarea(
            id="rq-textarea",
            placeholder="Enter research questions, one per line...",
            style={"width": "100%", "height": "150px"},
        ),
        html.Br(),
        dbc.Button("Run Analysis", id="run-analysis", color="primary"),
        html.Br(),
        html.Br(),
        html.Div(id="analysis-results"),
    ],
    fluid=True,
)

# Store uploaded data and metadata in dcc.Store components
app.layout.children += [
    dcc.Store(id="stored-data"),
    dcc.Store(id="stored-column-info"),
    dcc.Store(id="stored-output-paths"),
    dcc.Store(id="stored-rqs"),
    dcc.Store(id="stored-original-df"),
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


@app.callback(
    Output("upload-feedback", "children"), Input("upload-data", "contents"), State("upload-data", "filename")
)
def handle_upload(contents, filename):
    """Display the name of the uploaded file if the upload is successful"""
    if contents is None:
        return "", None

    return f"Uploaded file: {filename}"


@app.callback(
    Output("column-selectors", "children"),
    Input("upload-data", "contents"),
)
def show_column_selectors(contents):
    """Once a file has been uploaded, display dropdown menus
    where the user can identify which column contains the conversation ID, which column contains
    text etc.
    """
    if not contents:
        return "", None

    df = read_data(contents)

    options = [{"label": col, "value": col} for col in df.columns]
    return (
        html.Div(
            [
                html.H5("Step 1: Select relevant columns"),
                dbc.Row(
                    [
                        dbc.Col(dcc.Dropdown(id="conv-id-col", options=options, placeholder="Conversation ID column")),
                        dbc.Col(dcc.Dropdown(id="role-col", options=options, placeholder="Role column")),
                        dbc.Col(dcc.Dropdown(id="text-col", options=options, placeholder="Text column")),
                        dbc.Col(
                            dcc.Dropdown(
                                id="uuid-col",
                                options=[{"label": "None", "value": "None"}] + options,
                                placeholder="Unique ID (optional)",
                            )
                        ),
                    ]
                ),
            ]
        ),
    )


@app.callback(
    Output("stored-column-info", "data"),
    Input("conv-id-col", "value"),
    Input("role-col", "value"),
    Input("text-col", "value"),
    Input("uuid-col", "value"),
    State("stored-data", "data"),
)
def store_column_selection(conv_id, role_col, text_col, uuid_col, data_json):
    """
    Once columns have been selected, store the names of these columns so that we can
    use these later on with the data from the csv file.
    """

    if not all([conv_id, role_col, text_col]):
        stored_cols = None
    else:
        stored_cols = {
            "conv_id": conv_id,
            "role_col": role_col,
            "text_col": text_col,
            "uuid_col": uuid_col,
        }

    print(stored_cols)

    return stored_cols


@app.callback(
    Output("analysis-results", "children"),
    Output("stored-output-paths", "data"),
    Output("stored-rqs", "data"),
    Input("run-analysis", "n_clicks"),
    Input("upload-data", "contents"),
    State("stored-column-info", "data"),
    State("rq-textarea", "value"),
)
def run_analysis(n_clicks, contents, column_info, rq_text):
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
    output_paths = run_batch_check(conversation_dict, prompt_dict, OUTPUT_DIR)

    return dbc.Alert("LLM processing complete! See below for results.", color="success"), output_paths, rq_dict


@app.callback(
    Output("analysis-results", "children", allow_duplicate=True),
    Input("stored-output-paths", "data"),
    State("stored-rqs", "data"),
    prevent_initial_call="initial_duplicate",
)
def display_results(output_paths, rq_dict):
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


@app.callback(
    Output("quote-modal", "is_open"),
    Output("modal-body", "children"),
    Input({"type": "quote", "index": ALL}, "n_clicks"),
    State("stored-output-paths", "data"),
    State("stored-rqs", "data"),
    State("upload-data", "contents"),
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


if __name__ == "__main__":
    app.run(debug=True)
