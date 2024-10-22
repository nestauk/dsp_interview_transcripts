import pandas as pd
import pytest

from dsp_interview_transcripts.pipeline.archive.process_data import concatenate_consecutive_roles


def test_concatenate_consecutive_roles():
    # Sample made-up conversation data
    df = pd.DataFrame({
        'uuid': [
            'uuid1', 'uuid2', 'uuid3', 'uuid4',
            'uuid5', 'uuid6', 'uuid7', 'uuid8',
            'uuid9', 'uuid10'
        ],
        'timestamp': [
            '2024-05-01 10:00:00', '2024-05-01 10:01:00', '2024-05-01 10:02:00', 
            '2024-05-01 10:03:00', '2024-05-01 10:04:00', '2024-05-01 10:05:00', 
            '2024-05-01 10:06:00', '2024-05-01 10:07:00', '2024-05-01 10:08:00', 
            '2024-05-01 10:09:00'
        ],
        'conversation': [
            'conv1', 'conv1', 'conv1', 'conv1', 
            'conv1', 'conv1', 'conv1', 'conv1', 
            'conv1', 'conv1'
        ],
        'role': [
            'USER', 'USER', 'BOT', 'BOT', 
            'USER', 'BOT', 'USER', 'USER', 
            'BOT', 'USER'
        ],
        'text_clean': [
            "Hi!",
            "Can you help me?", 
            "Sure, how can I assist?",
            "Do you need more details?", 
            "Yes, I need help with my account.", 
            "What exactly seems to be the issue?", 
            "I forgot my password.",
            "Also, I can't access my email.", 
            "Let me help you with that.", "Thank you!"
        ]
    })

    # Expected result after concatenating consecutive roles
    expected_df = pd.DataFrame({
        'uuid': [
            'uuid1', 'uuid3', 'uuid5', 'uuid6', 
            'uuid7', 'uuid9', 'uuid10'
        ],
        'timestamp': [
            '2024-05-01 10:00:00', '2024-05-01 10:02:00', 
            '2024-05-01 10:04:00', '2024-05-01 10:05:00', 
            '2024-05-01 10:06:00', '2024-05-01 10:08:00',
            '2024-05-01 10:09:00'
        ],
        'conversation': [
            'conv1', 'conv1', 'conv1', 'conv1', 
            'conv1', 'conv1',
            'conv1'
        ],
        'role': [
            'USER', 'BOT', 'USER', 'BOT', 
            'USER', 'BOT', 'USER'
        ],
        'text_clean': [
            "Hi! Can you help me?", 
            "Sure, how can I assist? Do you need more details?", 
            "Yes, I need help with my account.", 
            "What exactly seems to be the issue?", 
            "I forgot my password. Also, I can't access my email.", 
            "Let me help you with that.",
            "Thank you!"
            
        ]
    })[['conversation', 'timestamp', 'text_clean', 'role', 'uuid']]

    # Call the function to test
    result_df = concatenate_consecutive_roles(df)
    print(result_df)

    # Assert that the result matches the expected dataframe
    pd.testing.assert_frame_equal(result_df, expected_df)
