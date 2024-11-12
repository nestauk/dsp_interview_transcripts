"""
Usage:

To run in test mode:
```
python dsp_interview_transcripts/pipeline/process_data.py
```

To run in production mode, add the flag `production`:
```
python dsp_interview_transcripts/pipeline/process_data.py production
```
"""

import random

from typing import List
from typing import Tuple

import nltk
import numpy as np
import pandas as pd
import plac
import torch

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts import logger
from dsp_interview_transcripts.getters.data_getters import save_to_s3
from dsp_interview_transcripts.getters.raw import get_raw_transcripts
from dsp_interview_transcripts.utils.data_cleaning import add_text_length
from dsp_interview_transcripts.utils.data_cleaning import clean_data
from dsp_interview_transcripts.utils.data_cleaning import convert_timestamp
from dsp_interview_transcripts.utils.data_cleaning import remove_preamble


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")

stop_words = set(stopwords.words("english"))

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# PyTorch seed (used by SentenceTransformer)
torch.manual_seed(RANDOM_SEED)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

QUESTIONS = config["questions"]


def concatenate_consecutive_roles(
    df: pd.DataFrame, text_col: str = "text_clean", conversation_col: str = "conversation", role_col: str = "role"
) -> pd.DataFrame:
    """
    Concatenates consecutive rows with the same role within a conversation. So if a user sends multiple short messages
    in succession, these get turned into one larger message.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the conversation data.
    - text_col (str): Name of the text column to concatenate.
    - conversation_col (str): Name of the column identifying conversation groups.
    - role_col (str): Name of the column identifying roles within the conversation.

    Returns:
    - pd.DataFrame: DataFrame with concatenated text for consecutive roles.
    """
    # Sort the dataframe to ensure correct order
    df = df.sort_values(by=[conversation_col, "timestamp"]).reset_index(drop=True)
    logger.info(f"Number of turns before concatenating consecutive roles: {len(df)}")

    # Create a mask to identify where the role changes or a new conversation starts
    df["role_change"] = (df[conversation_col] != df[conversation_col].shift(1)) | (
        df[role_col] != df[role_col].shift(1)
    )

    # Assign group numbers to consecutive rows with the same role within the same conversation
    df["turn"] = df["role_change"].cumsum()

    # Group by 'conversation' and 'group' to concatenate text
    grouped = (
        df.groupby([conversation_col, "turn"])
        .agg({"timestamp": "first", text_col: " ".join, role_col: "first", "uuid": "first"})
        .reset_index()
    )

    logger.info(f"Number of turns after concatenating consecutive roles: {len(grouped)}")

    grouped = grouped.drop(columns=["turn"])

    return grouped


def process_bot_qs(interviews_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """
    Produces a list of unique sentences produced by the bot.
    These will be matched to the interview guide.

    Parameters:
    - interviews_df (pd.DataFrame): DataFrame with bot conversation data.

    Returns:
    - Tuple[List[str], pd.DataFrame]: A list of unique sentences and DataFrame with exploded sentences.
    """
    bot_qs = interviews_df[interviews_df["role"] == "BOT"].copy()
    # split into individual sentences so that if the original question is contained within the utterance,
    # we have a better chance of catching it
    bot_qs["sentences"] = bot_qs["text_clean"].apply(lambda x: sent_tokenize(x))
    bot_qs = bot_qs.explode("sentences")
    bot_qs_list = bot_qs["sentences"].unique().tolist()
    return bot_qs_list, bot_qs


def match_questions(bot_qs_list: List[str], questions: List[str], threshold: float = 0.85) -> pd.DataFrame:
    """Match the input questions from our interview template,
    and the actual questions produced by the bot, using cosine similarity.

    Parameters:
    - bot_qs_list (List[str]): List of bot questions.
    - questions (List[str]): List of input questions to match against.
    - threshold (float): Similarity threshold for considering a match.

    Returns:
    - pd.DataFrame: DataFrame containing matched questions above the threshold.
    """
    bot_qs_embeddings = SENTENCE_MODEL.encode(bot_qs_list)
    input_qs_embeddings = SENTENCE_MODEL.encode(questions)

    similarities = cosine_similarity(
        bot_qs_embeddings,
        input_qs_embeddings,
    )

    # Find the index of the highest cosine similarity for each n-gram/lookup phrase combination
    max_indices = np.argmax(similarities, axis=1)

    # Retrieve the text of the corresponding target phrases
    most_similar_phrases = [questions[index] for index in max_indices]

    most_similar_similarities = [similarities[i, index] for i, index in enumerate(max_indices)]

    most_similar_pairs = list(zip(bot_qs_list, most_similar_phrases, most_similar_similarities))

    matches = pd.DataFrame(most_similar_pairs, columns=["bot_q", "question", "cosine_similarity"])

    final_matches = matches[
        matches["cosine_similarity"] > threshold
    ]  # temporary threshold until we've done some proper evaluation

    return final_matches


def get_best_matches(bot_qs: pd.DataFrame, final_matches: pd.DataFrame, questions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves original bot utterances and their best matching questions.

    Parameters:
    - bot_qs (pd.DataFrame): DataFrame of bot questions split into sentences.
    - final_matches (pd.DataFrame): DataFrame of matched questions.
    - questions_df (pd.DataFrame): DataFrame containing question details.

    Returns:
    - pd.DataFrame: DataFrame with highest similarity match for each bot utterance.
    """
    questions_matched = pd.merge(bot_qs, final_matches, left_on="sentences", right_on="bot_q", how="inner")
    # merge in the df that has the question number
    questions_matched = pd.merge(questions_matched, questions_df, left_on="question", right_on="question", how="left")

    uuid_question_counts = questions_matched.groupby("uuid")["question"].nunique()
    logger.info(
        f"The following utterances match to more than one question: {uuid_question_counts[uuid_question_counts > 1]}"
    )

    # Group by 'uuid' and keep the row with the highest 'cosine_similarity'
    questions_highest_similarity = questions_matched.loc[
        questions_matched.groupby("uuid")["cosine_similarity"].idxmax()
    ]
    return questions_highest_similarity


def get_sentiment(texts: List[str]) -> List[str]:
    """
    Predicts sentiment of texts using a pre-trained RoBERTa sentiment model.

    Parameters:
    - texts (List[str]): List of texts to analyze for sentiment.

    Returns:
    - List[str]: List of predicted sentiment labels ('Negative', 'Neutral', 'Positive') for each text.
    """
    roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    labels = ["Negative", "Neutral", "Positive"]

    # Tokenize all texts at once
    encoded_texts = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Pass all encoded texts through the model at once
    with torch.no_grad():  # Disable gradient computation for faster inference
        output = model(**encoded_texts)

    # Apply softmax to the scores
    scores = output.logits.detach().numpy()
    probabilities = softmax(scores, axis=1)

    # Get the label with the highest probability for each text
    predicted_labels = []
    for prob in probabilities:
        max_index = prob.argmax()  # Get the index of the highest probability
        predicted_labels.append(labels[max_index])  # Get the corresponding label

    return predicted_labels


def create_context(row: pd.Series, df: pd.DataFrame) -> str:
    """
    This is used to create an additional column in the dataframe that contains the preceding BOT > USER > BOT
    sequence before each USER message.

    You should first use concatenate_consecutive_roles() to make sure
    that the conversation is dyadic - otherwise you will just get the preceding 3 rows of data.

    Parameters:
    - row (pd.Series): Current row from the DataFrame to generate context for.
    - df (pd.DataFrame): DataFrame containing the conversation data.

    Returns:
    - str: Concatenated text of previous entries as context.
    """
    # Only proceed if the row is a USER entry
    if row["role"] == "USER":
        idx = row.name  # Current row index
        if idx >= 3:  # We need at least 3 previous rows to build context
            prev_rows = df.iloc[idx - 3 : idx]  # Take previous 3 rows
        else:
            prev_rows = df.iloc[:idx]

        return " | ".join(prev_rows["text_clean"])


def main(production: bool = False):

    MIN_LEN = config["min_length"]
    if production:
        DATA_OUT_PATH = config["prod_paths"]["interim_processed_data_s3_path"]
    else:
        DATA_OUT_PATH = config["test_paths"]["interim_processed_data_s3_path"]
    DATA_OUT_PATH = DATA_OUT_PATH.format(MIN_LEN=MIN_LEN)

    interviews_df = get_raw_transcripts()
    interviews_df = clean_data(interviews_df)

    logger.info(f"Number of interviews: {len(interviews_df['conversation'].unique())}")

    interviews_cleaned_df = (
        interviews_df
        # Make sure the conversations are sorted by time, so that the replies go in the right order
        .assign(timestamp_clean=lambda df: df["timestamp"].apply(convert_timestamp))
        .groupby("conversation", group_keys=False)
        .apply(lambda x: x.sort_values("timestamp_clean"))
        # Remove everything up to when bot asks if the instructions are clear - everything before is just noise
        .pipe(lambda df: df.groupby("conversation").apply(remove_preamble).reset_index(drop=True))
        # Group together consecutive responses by the same role
        .pipe(concatenate_consecutive_roles)
    )

    questions_df = pd.DataFrame(enumerate(QUESTIONS), columns=["q_number", "question"])

    bot_qs_list, bot_qs = process_bot_qs(interviews_cleaned_df)

    final_matches = match_questions(bot_qs_list, QUESTIONS)

    questions_highest_similarity = get_best_matches(bot_qs, final_matches, questions_df)

    # Merge back into the original df
    interviews_cleaned_df = pd.merge(
        interviews_cleaned_df,
        questions_highest_similarity[["uuid", "question", "q_number", "cosine_similarity"]],
        on="uuid",
        how="left",
    )

    # Forward fill the matched questions and their question numbers
    interviews_q_filled = (
        interviews_cleaned_df.copy()
        .assign(
            question=lambda df: df.groupby("conversation")["question"].ffill(),
            q_number=lambda df: df.groupby("conversation")["q_number"].ffill(),
        )
        .pipe(add_text_length)
        .assign(context=lambda df: df.apply(create_context, df=df, axis=1))
    )

    user_messages = interviews_q_filled[
        (interviews_q_filled["role"] == "USER") & (interviews_q_filled["text_length"] > MIN_LEN)
    ]

    # Get sentiments!!
    texts = user_messages["text_clean"].tolist()
    sentiments = get_sentiment(texts)

    user_messages["sentiment"] = sentiments

    logger.info(user_messages["sentiment"].value_counts())

    logger.info(f"Number of user messages: {len(user_messages)}")

    logger.info("Saving data...")
    save_to_s3(S3_BUCKET, user_messages, DATA_OUT_PATH)

    logger.info("Done!")


if __name__ == "__main__":
    plac.call(main)
