import base64
import json
import os

from io import StringIO
from pathlib import Path

import pandas as pd

from dash.exceptions import PreventUpdate

from dsp_interview_transcripts import PROJECT_DIR


def get_or_create_output_dir(session_id: str, test_mode: bool = False) -> str:
    """
    Returns the output directory for a session, creating it if needed.

    Args:
        session_id (str): Unique identifier for the session.
        test_mode (bool, optional): If True, returns a mock output directory for testing. Defaults to False.

    Returns:
        str: Path to the output directory as a string.
    """
    output_dir = PROJECT_DIR / "outputs" / ("mock_outputs" if test_mode else session_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def read_data(contents: str) -> pd.DataFrame:
    """
    If you have used an upload box to pick a csv file, you will
    need this logic to read in the csv as a pandas dataframe (Dash can't pass pandas dataframes between callbacks)

    Args:
        contents (str): Base64-encoded string from Dash upload component.

    Returns:
        pd.DataFrame: DataFrame created by reading the uploaded CSV file.
    """
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(StringIO(decoded.decode("utf-8")))
    return df


def get_cleaned_data(session_id: str) -> pd.DataFrame:
    """
    Loads cleaned data for a given session from the expected output directory.

    The page `upload.py` allows the user to upload a CSV file and select the columns they want to use.
    Then this dataframe gets passed through a cleaning function. We save the cleaned file and then this
    cleaned file gets used as the starting point for both the topic modelling page and the LLM analysis page.

    Args:
        session_id (str): Unique identifier for the session.

    Raises:
        PreventUpdate: If the cleaned data file does not exist.

    Returns:
        pd.DataFrame: DataFrame containing the cleaned data.
    """
    output_dir = get_or_create_output_dir(session_id, test_mode=False)

    cleaned_path = f"{output_dir}/cleaned_data.csv"

    if not os.path.exists(cleaned_path):
        raise PreventUpdate

    return pd.read_csv(cleaned_path)


def get_data_for_topic_modelling(session_id: str) -> pd.DataFrame:

    output_dir = get_or_create_output_dir(session_id, test_mode=False)

    combined_path = os.path.join(output_dir, "combined_data.csv")
    jsonl_path = os.path.join(output_dir, "raw_text_files.jsonl")

    if os.path.exists(combined_path):
        return pd.read_csv(combined_path)

    elif os.path.exists(jsonl_path):
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if isinstance(record, dict) and "filename" in record and "text" in record:
                    records.append(record)
        return pd.DataFrame(records)

    else:
        raise PreventUpdate
