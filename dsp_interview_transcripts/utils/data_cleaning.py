import emoji
import ftfy
import pandas as pd
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
small_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Embed the target sentence
TARGET_SENTENCE = "Are these instructions clear or do you need any further clarification?"
target_embedding = small_model.encode([TARGET_SENTENCE])

def remove_preamble(df, target_embedding=target_embedding, model=small_model):
    """Get rid of everything up until the bot asks if the instructions are clear
    """
    # Filter BOT messages
    bot_messages = df[df['role'] == 'BOT']
    
    # Embed BOT messages
    bot_embeddings = model.encode(bot_messages['text_clean'].tolist())
    
    # Calculate cosine similarity
    similarities = cosine_similarity(target_embedding, bot_embeddings).flatten()
    
    # Find the index of the most similar BOT message
    most_similar_idx = similarities.argmax()
    
    # Get the timestamp of that message
    cutoff_timestamp = bot_messages.iloc[most_similar_idx]['timestamp_clean']
    
    # Filter out messages prior to the cutoff timestamp
    return df[df['timestamp_clean'] > cutoff_timestamp]

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
