
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

import nltk

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download("punkt")
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

import numpy as np
from sentence_transformers import SentenceTransformer

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# PyTorch seed (used by SentenceTransformer)
import torch
torch.manual_seed(RANDOM_SEED)

from dsp_interview_transcripts import PROJECT_DIR, logger
from dsp_interview_transcripts.utils.data_cleaning import clean_data, convert_timestamp, add_text_length, remove_preamble

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DATA_PATH = PROJECT_DIR / "data/qual_af_transcripts.csv"

QUESTIONS = ["What, if anything, do you know about the Boiler Upgrade Scheme? If you don't know anything about the scheme, just give it your best guess.",
             "What, if anything, do you know about the process of applying for Boiler Upgrade Scheme funding?",
             "What do you think are the eligibility requirements for someone to use this scheme?",
             "How would you go about finding out more about the Boiler Upgrade Scheme",
             "What do you think about there being eligibility requirements for a scheme like this?",
             "As a homeowner, where do you see yourself in relation to the eligibility requirements?",
             "What, if any, type of work do you think needs to be done to a house to replace fossil fuel heating systems?",
             "What types of home upgrades would you consider getting done to your house to improve the efficiency of your heating system?",
             "What types of work to your house wouldn't you consider?",
             "What are some energy-efficient heating systems that you could consider, apart from the one currently in use at your home?",
             "Is there anything we've talked about you'd like to discuss further?"]

MIN_LEN = 9

def concatenate_consecutive_roles(df, text_col='text_clean', conversation_col='conversation', role_col='role'):
    """Concatenates consecutive rows with the same role within a conversation."""
    # Sort the dataframe to ensure correct order (if it's not already sorted)
    df = df.sort_values(by=[conversation_col, 'timestamp']).reset_index(drop=True)
    logger.info(f"Number of turns before concatenating consecutive roles: {len(df)}")

    # Create a mask to identify where the role changes or a new conversation starts
    df['role_change'] = (df[conversation_col] != df[conversation_col].shift(1)) | (df[role_col] != df[role_col].shift(1))

    # Assign group numbers to consecutive rows with the same role within the same conversation
    df['turn'] = df['role_change'].cumsum()

    # Group by 'conversation' and 'group' to concatenate text
    grouped = df.groupby([conversation_col, 'turn']).agg({
        'timestamp': 'first',
        text_col: ' '.join,
        role_col: 'first',
        'uuid': 'first'
    }).reset_index()
    
    logger.info(f"Number of turns after concatenating consecutive roles: {len(grouped)}")

    grouped = grouped.drop(columns=['turn'])

    return grouped

def create_q_and_a_column(data_df, text_col = 'text_clean', conversation_col='conversation'):
    """Concatenates each user message with the immediately preceding bot message.
    In cases where the user has given a really short response, this should provide some
    helpful context.
    """
    # Initialize a list to store the Q&A
    q_and_a_list = []
    
    # Variable to keep track of the most recent BOT text
    last_bot_text = ""
    
    df = data_df.copy()
    
    # Track the current conversation ID to know when it changes
    current_conversation = None
    
    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        
        # Check if the conversation has changed
        if current_conversation != row[conversation_col]:
            current_conversation = row[conversation_col]
            last_bot_text = ""  # Reset the last bot text for a new conversation
        
        if row['role'] == 'BOT':
            # Update the last bot text
            last_bot_text = row[text_col]
            q_and_a_list.append('')  # No Q&A for BOT rows
        elif row['role'] == 'USER':
            # Combine the last bot text and current user text if there is a preceding bot text,
            # otherwise just use the user text
            q_and_a_list.append(f"{last_bot_text}\n{row[text_col]}" if last_bot_text else row[text_col])
        else:
            # In case there's any other role, we leave it empty
            q_and_a_list.append('')

    # Add the new 'q_and_a' column to the DataFrame
    df['q_and_a'] = q_and_a_list

    return df

def process_bot_qs(interviews_df: pd.DataFrame) -> list:
    """Produce a list of each unique sentence produced by the bot
    """
    bot_qs = interviews_df[interviews_df['role'] == 'BOT']
    # split into individual sentences so that if the original question is contained within the utterance,
    # we have a better chance of catching it
    bot_qs["sentences"] = bot_qs['text_clean'].apply(lambda x: sent_tokenize(x))
    bot_qs = bot_qs.explode("sentences")
    bot_qs_list = bot_qs['sentences'].unique().tolist()
    return bot_qs_list, bot_qs

def match_questions(bot_qs_list, questions, threshold=0.85):
    """Find the best matches between the input questions from our interview template,
    and the actual questions produced by the bot.
    """
    bot_qs_embeddings = SENTENCE_MODEL.encode(bot_qs_list)
    input_qs_embeddings = SENTENCE_MODEL.encode(questions)

    similarities = cosine_similarity(bot_qs_embeddings, input_qs_embeddings, )
    
    # Find the index of the highest cosine similarity for each n-gram/lookup phrase combination
    max_indices = np.argmax(similarities, axis=1)

    # Retrieve the text of the corresponding target phrases
    most_similar_phrases = [questions[index] for index in max_indices]

    most_similar_similarities = [
                similarities[i, index] for i, index in enumerate(max_indices)
            ]

    most_similar_pairs = list(
                zip(bot_qs_list, most_similar_phrases, most_similar_similarities)
            )

    matches = pd.DataFrame(
                most_similar_pairs, columns=["bot_q", "question", "cosine_similarity"]
            )
    
    final_matches = matches[matches['cosine_similarity'] > threshold] # temporary threshold until we've done some proper evaluation
    
    return final_matches

def get_best_matches(bot_qs, final_matches, questions_df):
    """
    Create a dataframe of the original bot utterances and the questions they were matched to
    """
    questions_matched = pd.merge(bot_qs, final_matches, left_on='sentences', right_on='bot_q', how='inner')
    # merge in the df that has the question number
    questions_matched = pd.merge(questions_matched, questions_df, left_on='question', right_on='question', how='left')
    
    uuid_question_counts = questions_matched.groupby('uuid')['question'].nunique()
    logger.info(f"The following utterances match to more than once question: {uuid_question_counts[uuid_question_counts > 1]}")
    
    # Group by 'uuid' and keep the row with the highest 'cosine_similarity'
    questions_highest_similarity = questions_matched.loc[questions_matched.groupby('uuid')['cosine_similarity'].idxmax()]
    return questions_highest_similarity

def get_sentiment(texts):
    roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    
    labels = ['Negative', 'Neutral', 'Positive']

    # Tokenize all texts at once
    encoded_texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    
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

def create_context(row, df):
    # Only proceed if the row is a USER entry
    if row['role'] == 'USER':
        idx = row.name  # Current row index
        if idx >= 3:  # We need at least 3 previous rows to build context
            prev_rows = df.iloc[idx-3:idx]  # Take previous 3 rows
        else:
            prev_rows = df.iloc[:idx]
        
        return ' | '.join(prev_rows['text_clean'])

if __name__ == "__main__":
    interviews_df = pd.read_csv(DATA_PATH)
    interviews_df = clean_data(interviews_df)
        
    logger.info(f"Number of interviews: {len(interviews_df['conversation'].unique())}")
        
    # Make sure the conversations are sorted by time, so that the replies go in the right order
    interviews_df['timestamp_clean'] = interviews_df['timestamp'].apply(convert_timestamp)
    interviews_df = interviews_df.groupby('conversation', group_keys=False).apply(lambda x: x.sort_values('timestamp_clean'))
        
    # Remove everything up to when bot asks if the instructions are clear - everything before is just noise
    interviews_cleaned_df = interviews_df.groupby('conversation').apply(remove_preamble).reset_index(drop=True)
        
    # Group together consecutive responses by the same role
    interviews_df = concatenate_consecutive_roles(interviews_df)
        
    questions_df = pd.DataFrame(enumerate(QUESTIONS), columns=["q_number", "question"])
        
    bot_qs_list, bot_qs = process_bot_qs(interviews_df)
        
    final_matches = match_questions(bot_qs_list, QUESTIONS)
        
    questions_highest_similarity = get_best_matches(bot_qs, final_matches, questions_df)
        
    # Merge back into the original df
    interviews_df = pd.merge(interviews_df, questions_highest_similarity[['uuid', 'question', 'q_number', 'cosine_similarity']], on='uuid', how='left')
    
    # Forward fill the matched questions and their question numbers
    interviews_q_filled = interviews_df.copy()
    interviews_q_filled['question'] = interviews_q_filled.groupby('conversation')['question'].ffill()
    interviews_q_filled['q_number'] = interviews_q_filled.groupby('conversation')['q_number'].ffill()
    
    interviews_q_filled = add_text_length(interviews_q_filled)
    
    interviews_q_filled['context'] = interviews_q_filled.apply(create_context, df=interviews_q_filled, axis=1)
    
    user_messages = interviews_q_filled[(interviews_q_filled['role']=='USER')& (interviews_q_filled['text_length'] > MIN_LEN)]
    
    # Get sentiments!!
    texts = user_messages['text_clean'].tolist()
    sentiments = get_sentiment(texts)
    
    user_messages['sentiment'] = sentiments
    
    logger.info(user_messages['sentiment'].value_counts())
    
    logger.info(f"Number of user messages: {len(user_messages)}")
    
    logger.info("Saving data...")
    user_messages.to_csv(PROJECT_DIR / f"data/user_messages_min_len_{MIN_LEN}_w_sentiment.csv", index=False)
    
    logger.info("Done!")