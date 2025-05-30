import base64
import datetime
import io
import json
import mimetypes
import os
import uuid

import emoji
import ftfy
import pandas as pd

from dash import Input
from dash import Output
from dash import State
from dash.exceptions import PreventUpdate
from docx import Document

from utils.dash_utils import get_or_create_output_dir


def replace_punct(text: str) -> str:
    """
    Replaces or removes specific punctuation characters.

    Args:
        text (str): The text string to process.

    Returns:
        str: The cleaned text with replacements made.
    """
    if not isinstance(text, str):
        return ""

    text = (
        text.replace("&", "and")
        .replace("\xa0", " ")
        .replace("\r", ".")
        .replace("\n", ".")
        .replace("[", "")
        .replace("]", "")
    )

    return text.strip()


def clean_data(data_df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    """
    Does minimal text cleaning on *tabular* data

    Args:
        data_df (pd.DataFrame): DataFrame containing conversation data, with at least 'text' and 'text_clean' columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with processed text.
    """

    data_df[text_col] = data_df[text_col].fillna("")

    data_df = data_df[data_df[text_col].astype(str).str.strip() != ""]

    # Fix improperly coded characters
    data_df["text_clean"] = data_df[text_col].apply(lambda x: ftfy.fix_text(x))

    # Remove emojis
    data_df["text_clean"] = data_df["text_clean"].apply(lambda x: emoji.demojize(x))

    # Replace punctuation
    data_df["text_clean"] = data_df["text_clean"].apply(replace_punct)

    return data_df


def extract_text_from_docx(decoded_bytes):
    doc = Document(io.BytesIO(decoded_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(decoded_bytes):
    return decoded_bytes.decode("utf-8")


def extract_text_from_vtt(decoded_bytes):
    lines = decoded_bytes.decode("utf-8").splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line == "" or "-->" in line or line.isdigit():
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


# def chunk_text_lines(text, chunk_size=4, overlap=2):
#     lines = [line.strip() for line in text.splitlines() if line.strip()]
#     chunks = []
#     for i in range(0, len(lines) - chunk_size + 1, chunk_size - overlap):
#         chunk = "\n".join(lines[i:i + chunk_size])
#         chunks.append(chunk)
#     return chunks


def make_session_id():
    timestamp_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp_prefix}_{uuid.uuid4()}"


def register_upload_callbacks(app):
    @app.callback(
        Output("upload-feedback", "children"),
        Output("data-store", "data"),
        Output("session-id", "data"),
        Output("column-store", "data"),
        Output("save-feedback", "children"),
        Output("conv-id-dropdown", "value"),
        Output("role-dropdown", "value"),
        Output("text-dropdown", "value"),
        Output("uuid-dropdown", "value"),
        Output("show-column-select-flag", "data"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def handle_upload(contents_list, filenames, stored_sid):
        if not contents_list or not filenames:
            raise PreventUpdate

        session_id = stored_sid or make_session_id()
        output_dir = get_or_create_output_dir(session_id)

        tabular_dfs = []
        jsonl_entries = []
        tabular = False

        for contents, filename in zip(contents_list, filenames):
            content_string = contents.split(",")[1]
            decoded = base64.b64decode(content_string)

            if filename.endswith(".csv"):
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
                df["source_file"] = filename
                tabular_dfs.append(df)
                tabular = True

            elif filename.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(decoded))
                df["source_file"] = filename
                tabular_dfs.append(df)
                tabular = True

            elif filename.endswith(".txt"):
                text = extract_text_from_txt(decoded)
                jsonl_entries.append({"filename": filename, "text": text})

            elif filename.endswith(".docx"):
                text = extract_text_from_docx(decoded)
                jsonl_entries.append({"filename": filename, "text": text})

            elif filename.endswith(".vtt"):
                text = extract_text_from_vtt(decoded)
                jsonl_entries.append({"filename": filename, "text": text})

            else:
                continue

        # Save text data to .jsonl if applicable
        if jsonl_entries:
            jsonl_path = os.path.join(output_dir, "raw_text_files.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for entry in jsonl_entries:
                    json.dump(entry, f)
                    f.write("\n")

        # Save tabular data to CSV if applicable
        if tabular_dfs:
            combined_df = pd.concat(tabular_dfs, ignore_index=True)
            combined_df.to_csv(os.path.join(output_dir, "combined_data.csv"), index=False)
        else:
            combined_df = pd.DataFrame()

        if combined_df.empty and not jsonl_entries:
            return ("❌ No valid files uploaded", None, None, None, "", None, None, None, None, False)

        feedback = f"✅ Uploaded {len(filenames)} files"

        return (
            feedback,
            combined_df.to_json(date_format="iso", orient="split") if not combined_df.empty else None,
            session_id,
            None,
            "",
            None,
            None,
            None,
            None,
            tabular,
        )

    @app.callback(
        Output("column-section", "style"),
        Output("conv-id-dropdown", "options"),
        Output("role-dropdown", "options"),
        Output("text-dropdown", "options"),
        Output("uuid-dropdown", "options"),
        Input("data-store", "data"),
        Input("show-column-select-flag", "data"),
    )
    def show_column_selector(jsonified_data, show_column_select):
        if not jsonified_data or not show_column_select:
            return {"display": "none"}, [], [], [], []

        df = pd.read_json(jsonified_data, orient="split")
        opts = [{"label": col, "value": col} for col in df.columns]
        return {"display": "block"}, opts, opts, opts, opts

    @app.callback(
        Output("save-feedback", "children", allow_duplicate=True),
        Output("column-store", "data", allow_duplicate=True),
        Input("save-btn", "n_clicks"),
        State("data-store", "data"),
        State("conv-id-dropdown", "value"),
        State("role-dropdown", "value"),
        State("text-dropdown", "value"),
        State("uuid-dropdown", "value"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def save_columns_and_clean(n, json_data, conv_col, role_col, text_col, uuid_col, session_id):
        if not (n and json_data and role_col and text_col and session_id):
            raise PreventUpdate

        df = pd.read_json(json_data, orient="split")

        # If UUID column is missing or has nulls, generate UUIDs
        if uuid_col not in df.columns or df[uuid_col].isnull().any():
            df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
            uuid_col = "uuid"
        if conv_col is None:
            conv_col = "source_file"

        df[uuid_col] = df[uuid_col].astype(str)

        # Clean data
        cleaned_df = clean_data(df, text_col=text_col)

        # Save cleaned data
        output_dir = get_or_create_output_dir(session_id)
        cleaned_df.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)

        # Store columns mapping
        cols = {"conv_id": conv_col, "role": role_col, "text": text_col, "uuid": uuid_col}
        return ("✅ Cleaned data saved successfully!", cols)
