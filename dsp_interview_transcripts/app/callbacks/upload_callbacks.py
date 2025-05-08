import base64
import io
import os
import uuid

import pandas as pd

from dash import Input
from dash import Output
from dash import State
from dash.exceptions import PreventUpdate

from dsp_interview_transcripts.utils.data_cleaning import clean_data
from utils.dash_utils import get_or_create_output_dir


def register_upload_callbacks(app):

    # Callback to parse upload and reset state
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
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        State("session-id", "data"),
        prevent_initial_call=True,
    )
    def handle_upload(contents, filename, stored_sid):
        if not contents:
            raise PreventUpdate
        # Decode and read CSV
        content_string = contents.split(",")[1]
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        # Preserve or generate session ID
        session_id = stored_sid or str(uuid.uuid4())
        # Reset column selections and feedback
        feedback = f"Uploaded: {filename}"
        return (
            feedback,
            df.to_json(date_format="iso", orient="split"),
            session_id,
            None,  # clear column-store
            "",  # clear save-feedback
            None,
            None,
            None,
            None,  # reset dropdown values
        )

    # Callback to show column dropdowns once data is uploaded
    @app.callback(
        Output("column-section", "style"),
        Output("conv-id-dropdown", "options"),
        Output("role-dropdown", "options"),
        Output("text-dropdown", "options"),
        Output("uuid-dropdown", "options"),
        Input("data-store", "data"),
    )
    def show_column_selector(jsonified_data):
        if not jsonified_data:
            return {"display": "none"}, [], [], [], []
        df = pd.read_json(jsonified_data, orient="split")
        opts = [{"label": col, "value": col} for col in df.columns]
        return {"display": "block"}, opts, opts, opts, opts

    # Callback to save selected columns, clean and save data
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
        if not (n and json_data and conv_col and role_col and text_col and session_id):
            raise PreventUpdate
        df = pd.read_json(json_data, orient="split")
        # Ensure UUID column
        if uuid_col not in df.columns or df[uuid_col].isnull().any():
            df["uuid"] = [str(uuid.uuid4()) for _ in range(len(df))]
            uuid_col = "uuid"
        df[uuid_col] = df[uuid_col].astype(str)
        # Clean data
        cleaned_df = clean_data(df, text_col)
        # Save cleaned data
        output_dir = get_or_create_output_dir(session_id)
        cleaned_df.to_csv(os.path.join(output_dir, "cleaned_data.csv"), index=False)
        # Store columns mapping
        cols = {"conv_id": conv_col, "role": role_col, "text": text_col, "uuid": uuid_col}
        return ("✅ Columns saved and data cleaned!", cols)
