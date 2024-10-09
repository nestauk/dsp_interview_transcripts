"""
Tries to broadly break the interview into sections by which question they follow.
"""

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

from dsp_interview_transcripts import PROJECT_DIR, logger
from dsp_interview_transcripts.utils.data_cleaning import clean_data, convert_timestamp, add_text_length
from dsp_interview_transcripts.pipeline.process_data import concatenate_consecutive_roles

RANDOM_SEED = 42
DATA_PATH = PROJECT_DIR / "data/qual_af_transcripts.csv"

torch.manual_seed(RANDOM_SEED)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

QUESTIONS = ["What, if anything, do you know about the Boiler Upgrade Scheme? If you don't know anything about the scheme, just give it your best guess.",
             "What, if anything, do you know about the process of applying for Boiler Upgrade Scheme funding?",
             "What do you think are the eligibility requirements for someone to use this scheme?",
             "How would you go about finding out more about the Boiler Upgrade Scheme",
             "What do you think about there being eligiblity requirements for a scheme like this?",
             "As a homeowner, where do you see yourself in relation to the eligibility requirements?",
             "What, if any, type of work do you think needs to be done to a house to replace fossil fuel heating systems?",
             "What types of home upgrades would you consider getting done to your house to improve the efficiency of your heating system?",
             "What types of work to your house wouldn't you consider?",
             "What are some energy-efficient heating systems that you could consider, apart from the one currently in use at your home?",
             "Is there anything we've talked about you'd like to discuss further?"]

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

if __name__ == "__main__":
    interviews_df = pd.read_csv(DATA_PATH)
    interviews_df = clean_data(interviews_df)
        
    logger.info(f"Number of interviews: {len(interviews_df['conversation'].unique())}")
        
    # Make sure the conversations are sorted by time, so that the replies go in the right order
    interviews_df['timestamp_clean'] = interviews_df['timestamp'].apply(convert_timestamp)
    interviews_df = interviews_df.groupby('conversation', group_keys=False).apply(lambda x: x.sort_values('timestamp_clean'))
        
    # Group together consecutive responses by the same role
    interviews_df = concatenate_consecutive_roles(interviews_df)
    
    questions_df = pd.DataFrame(enumerate(QUESTIONS), columns=["q_number", "question"])
    
    bot_qs_list, bot_qs = process_bot_qs(interviews_df)
    
    final_matches = match_questions(bot_qs_list, QUESTIONS)
    
    questions_highest_similarity = get_best_matches(bot_qs, final_matches, questions_df)
    
    # Merge back into the original df
    interviews_df = pd.merge(interviews_df, questions_highest_similarity[['uuid', 'question', 'q_number']], on='uuid', how='left')
    
    # Forward fill the matched questions and their question numbers
    interviews_q_filled = interviews_df.copy()
    interviews_q_filled['question'] = interviews_q_filled.groupby('conversation')['question'].ffill()
    interviews_q_filled['q_number'] = interviews_q_filled.groupby('conversation')['q_number'].ffill()
    
    logger.info(f"Question counts: \n {interviews_q_filled['question'].value_counts()}")
    
    interviews_q_filled['q_number'] = interviews_q_filled['q_number'].replace({np.nan: -1})
    interviews_q_filled['q_number'] = interviews_q_filled['q_number'].astype(int)
    
    # Save a separate dataframe for user responses following each question
    for q in interviews_q_filled['q_number'].unique():
        interviews_q_filled[(interviews_q_filled['q_number'] == q) & (interviews_q_filled['role'] == 'USER')].to_csv(PROJECT_DIR / 'data/interview_q_{}.csv'.format(q), index=False)
