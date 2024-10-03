import emoji
import ftfy
import pandas as pd
import re

def convert_timestamp(timestamp):
    # Remove the daylight saving time '+01:00' and trailing whitespace
    cleaned_timestamp = re.sub('\+01[\:]?00$', '', timestamp)
    cleaned_timestamp = cleaned_timestamp.rstrip()
    try:
        # Convert to datetime
        return pd.to_datetime(cleaned_timestamp)
    except ValueError:
        # If conversion fails, print the problematic timestamp
        print(f"Cannot convert timestamp: {cleaned_timestamp}")
        return pd.NaT  # Return NaT (Not a Time) for invalid timestamps

def fill_text_with_transcript(data_df):
    """
    Fills the 'text' column with the 'transcript' column where 'text' is missing.
    """
    data_df = data_df.assign(text=lambda x: x["text"].fillna(x["transcript"]))
    return data_df.fillna({"text": ""})

def add_text_length(data_df, text_col='text_clean'):
    """
    Adds a new column 'text_length' with the length of the 'text' column.
    """
    data_df = data_df.assign(text_length=lambda x: x[text_col].apply(lambda text: len(text.split())))
    return data_df

def replace_punct(text):
    text = (
        text.replace("&", "and")
        .replace("\xa0", " ")
        .replace("\r", ".")
        .replace("\n", ".")
        .replace("[", "")
        .replace("]", "")
    )

    return text.strip()

def clean_data(data_df):
    """
    Pulls together all the previous cleaning steps
    """
    data_df = fill_text_with_transcript(data_df)
    
    # Fix improperly coded characters
    data_df['text_clean'] = data_df['text'].apply(lambda x: ftfy.fix_text(x))
    
    # Remove emojis
    data_df['text_clean'] = data_df['text_clean'].apply(lambda x: emoji.demojize(x))
    
    # Replace punctuation
    data_df['text_clean'] = data_df['text_clean'].apply(replace_punct)
    
    return data_df

