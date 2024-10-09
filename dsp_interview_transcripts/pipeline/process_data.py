import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR, logger
from dsp_interview_transcripts.utils.data_cleaning import clean_data, convert_timestamp, add_text_length

DATA_PATH = PROJECT_DIR / "data/qual_af_transcripts.csv"

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
        'timestamp': 'first',  # Keep the first timestamp in each group
        text_col: ' '.join,    # Concatenate the 'text_clean' column
        role_col: 'first',     # Keep the role (either BOT or USER)
        'uuid': 'first'        # Keep the first UUID in each group
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

if __name__ == "__main__":
    interviews_df = pd.read_csv(DATA_PATH)
    interviews_df = clean_data(interviews_df)
    
    logger.info(f"Number of interviews: {len(interviews_df['conversation'].unique())}")
    
    # Make sure the conversations are sorted by time, so that the replies go in the right order
    interviews_df['timestamp_clean'] = interviews_df['timestamp'].apply(convert_timestamp)
    interviews_df = interviews_df.groupby('conversation', group_keys=False).apply(lambda x: x.sort_values('timestamp_clean'))
    
    # Group together consecutive responses by the same role
    interviews_df = concatenate_consecutive_roles(interviews_df)
    
    interviews_df = create_q_and_a_column(interviews_df)
    
    interviews_df = add_text_length(interviews_df, 'text_clean')
    
    q_and_a_df = interviews_df[interviews_df['q_and_a'] != '']
    q_and_a_df.to_csv(PROJECT_DIR / "data/q_and_a.csv", index=False)
    
    user_messages = interviews_df[(interviews_df['role']=='USER') & (interviews_df['text_length'] >= 4)] # only answers at least 4 words long
    user_messages.to_csv(PROJECT_DIR / "data/user_messages.csv", index=False)
