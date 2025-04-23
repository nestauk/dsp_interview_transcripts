import re

from typing import Union

import emoji
import ftfy
import pandas as pd


def convert_timestamp(timestamp: str) -> Union[pd.Timestamp, pd.NaT]:
    """
    Converts a timestamp string to a pandas Timestamp object, removing any trailing timezone information.
    If the timestamp is invalid, returns NaT.

    Args:
        timestamp (str): The timestamp string to convert, potentially with timezone information.

    Returns:
        Union[pd.Timestamp, pd.NaT]: A pandas Timestamp object if conversion is successful; NaT otherwise.
    """
    # Remove the daylight saving time '+01:00' and trailing whitespace
    cleaned_timestamp = re.sub("\+01[\:]?00$", "", timestamp)
    cleaned_timestamp = cleaned_timestamp.rstrip()
    try:
        # Convert to datetime
        return pd.to_datetime(cleaned_timestamp)
    except ValueError:
        # If conversion fails, print the problematic timestamp
        print(f"Cannot convert timestamp: {cleaned_timestamp}")
        return pd.NaT  # Return NaT (Not a Time) for invalid timestamps


def fill_text_with_transcript(data_df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    """
    Fills missing values in the specified text column with corresponding values from the 'transcript' column,
    if it exists in the DataFrame.

    Args:
        data_df (pd.DataFrame): DataFrame containing the text and optionally a 'transcript' column.
        text_col (str): Name of the text column to fill.

    Returns:
        pd.DataFrame: Updated DataFrame with missing text values filled from 'transcript' if available.
    """
    if "transcript" in data_df.columns:
        data_df[text_col] = data_df[text_col].fillna(data_df["transcript"])

    # Fill any remaining NaNs - eg if both 'text' and 'transcript' were NaN
    data_df[text_col] = data_df[text_col].fillna("")

    return data_df


def add_text_length(data_df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """
    Adds a 'text_length' column to the DataFrame, representing the number of words in the text column.

    Args:
        data_df (pd.DataFrame): DataFrame containing a text column.
        text_col (str): The name of the column containing text data (default is 'text_clean').

    Returns:
        pd.DataFrame: Updated DataFrame with a new 'text_length' column.
    """
    data_df = data_df.assign(text_length=lambda x: x[text_col].apply(lambda text: len(text.split())))
    return data_df


def replace_punct(text: str) -> str:
    """
    Replaces or removes specific punctuation characters.

    Args:
        text (str): The text string to process.

    Returns:
        str: The cleaned text with replacements made.
    """
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
    Pulls together all the previous cleaning steps

    Args:
        data_df (pd.DataFrame): DataFrame containing conversation data, with at least 'text' and 'text_clean' columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with processed text.
    """
    data_df = fill_text_with_transcript(data_df)

    # Fix improperly coded characters
    data_df["text_clean"] = data_df[text_col].apply(lambda x: ftfy.fix_text(x))

    # Remove emojis
    data_df["text_clean"] = data_df["text_clean"].apply(lambda x: emoji.demojize(x))

    # Replace punctuation
    data_df["text_clean"] = data_df["text_clean"].apply(replace_punct)

    return data_df
