import pandas as pd
import pytest

from dsp_interview_transcripts.pipeline.archive.process_data import create_q_and_a_column, concatenate_consecutive_roles


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'conversation': [1, 1, 1, 2, 2, 2],
        'uuid': [1, 2, 3, 4, 5, 6],
        'timestamp': ['2024-05-01 10:00:00',
                      '2024-05-01 10:01:00',
                      '2024-05-01 10:02:00', 
            '2024-05-01 10:03:00',
            '2024-05-01 10:04:00', 
            '2024-05-01 10:05:00'],
        'role': ['BOT', 'USER', 'USER', 'BOT', 'USER', 'USER'],
        'text_clean': [
            "Hello, I am a bot", 
            "Hello", 
            "Are you going to ask me questions?", 
            "Hello, I am still a bot", 
            "Hi, I am human", 
            "Do you have questions for me?"
        ]
    })

def test_incorrect_behavior_with_consecutive_roles(sample_df):
    """Test that the function performs incorrectly with consecutive BOT or USER messages."""
    # Call the original function without preprocessing
    result_df = create_q_and_a_column(sample_df, text_col='text_clean', conversation_col='conversation')
    
    print(result_df['q_and_a'].tolist())
    # Expected incorrect output: Bot message followed by incorrect user messages
    incorrect_expected_q_and_a = [
        '',  # Bot message has no Q&A
        'Hello, I am a bot\nHello',  # First user response
        'Hello, I am a bot\nAre you going to ask me questions?',  # Second user response
        '',  # New conversation, first bot message
        "Hello, I am still a bot\nHi, I am human",  # First user response in new conversation
        'Hello, I am still a bot\nDo you have questions for me?'  # Second user response in new conversation
    ]

    # The function should incorrectly process consecutive user messages without proper reset
    assert result_df['q_and_a'].tolist() == incorrect_expected_q_and_a, "The function incorrectly processed consecutive USER messages."

def test_correct_behavior_after_preprocessing(sample_df):
    """Test that the function works correctly after ensuring alternating BOT and USER roles."""

    preprocessed_df = concatenate_consecutive_roles(sample_df)

    # Call the original function on the preprocessed DataFrame
    result_df = create_q_and_a_column(preprocessed_df, text_col='text_clean', conversation_col='conversation')
    print(result_df['q_and_a'].tolist())

    # Expected correct output after preprocessing
    expected_q_and_a = [
        '',  # Bot message has no Q&A
        'Hello, I am a bot\nHello Are you going to ask me questions?',
        '',  # New conversation, first bot message
        "Hello, I am still a bot\nHi, I am human Do you have questions for me?",  # First user response in new conversation
    ]

    # Check that the 'q_and_a' column was created and matches expected output after preprocessing
    assert 'q_and_a' in result_df.columns
    assert result_df['q_and_a'].tolist() == expected_q_and_a, "The function processed the Q&A correctly after preprocessing."

def test_empty_bot_text_reset_after_preprocessing(sample_df):
    """Test that the bot text resets at the end of each conversation after preprocessing."""
    # Preprocess the DataFrame to ensure alternating BOT and USER roles
    preprocessed_df = concatenate_consecutive_roles(sample_df)

    # Call the function after preprocessing
    result_df = create_q_and_a_column(preprocessed_df, text_col='text_clean', conversation_col='conversation')
    print(result_df['q_and_a'].tolist())

    # Ensure bot message was reset after conversation 1 ends
    assert result_df.loc[2, 'q_and_a'] == '', "The bot text was reset correctly after conversation 1."
