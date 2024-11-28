from unittest.mock import patch

import pandas as pd
import pytest

from dsp_interview_transcripts.pipeline.process_data import concatenate_consecutive_roles
from dsp_interview_transcripts.pipeline.process_data import create_context
from dsp_interview_transcripts.pipeline.process_data import get_best_matches
from dsp_interview_transcripts.pipeline.process_data import get_sentiment
from dsp_interview_transcripts.pipeline.process_data import match_questions
from dsp_interview_transcripts.pipeline.process_data import process_bot_qs


@pytest.fixture
def mock_dataframe():
    """Fixture for a mock DataFrame simulating a conversation structure."""
    data = {
        "conversation": [1, 1, 1, 2, 2],
        "text_clean": ["Hi", "Hello", "How are you?", "Good morning", "Good night"],
        "role": ["USER", "USER", "BOT", "USER", "BOT"],
        "timestamp": [
            "2021-01-01 10:00:00",
            "2021-01-01 10:01:00",
            "2021-01-01 10:02:00",
            "2021-01-02 10:00:00",
            "2021-01-02 10:01:00",
        ],
        "uuid": [1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_questions():
    return ["How are you?", "Good morning?", "Good night?"]


def test_concatenate_consecutive_roles():
    # Sample made-up conversation data
    df = pd.DataFrame(
        {
            "uuid": ["uuid1", "uuid2", "uuid3", "uuid4", "uuid5", "uuid6", "uuid7", "uuid8", "uuid9", "uuid10"],
            "timestamp": [
                "2024-05-01 10:00:00",
                "2024-05-01 10:01:00",
                "2024-05-01 10:02:00",
                "2024-05-01 10:03:00",
                "2024-05-01 10:04:00",
                "2024-05-01 10:05:00",
                "2024-05-01 10:06:00",
                "2024-05-01 10:07:00",
                "2024-05-01 10:08:00",
                "2024-05-01 10:09:00",
            ],
            "conversation": ["conv1", "conv1", "conv1", "conv1", "conv1", "conv1", "conv1", "conv1", "conv1", "conv1"],
            "role": ["USER", "USER", "BOT", "BOT", "USER", "BOT", "USER", "USER", "BOT", "USER"],
            "text_clean": [
                "Hi!",
                "Can you help me?",
                "Sure, how can I assist?",
                "Do you need more details?",
                "Yes, I need help with my account.",
                "What exactly seems to be the issue?",
                "I forgot my password.",
                "Also, I can't access my email.",
                "Let me help you with that.",
                "Thank you!",
            ],
        }
    )

    # Expected result after concatenating consecutive roles
    expected_df = pd.DataFrame(
        {
            "uuid": ["uuid1", "uuid3", "uuid5", "uuid6", "uuid7", "uuid9", "uuid10"],
            "timestamp": [
                "2024-05-01 10:00:00",
                "2024-05-01 10:02:00",
                "2024-05-01 10:04:00",
                "2024-05-01 10:05:00",
                "2024-05-01 10:06:00",
                "2024-05-01 10:08:00",
                "2024-05-01 10:09:00",
            ],
            "conversation": ["conv1", "conv1", "conv1", "conv1", "conv1", "conv1", "conv1"],
            "role": ["USER", "BOT", "USER", "BOT", "USER", "BOT", "USER"],
            "text_clean": [
                "Hi! Can you help me?",
                "Sure, how can I assist? Do you need more details?",
                "Yes, I need help with my account.",
                "What exactly seems to be the issue?",
                "I forgot my password. Also, I can't access my email.",
                "Let me help you with that.",
                "Thank you!",
            ],
        }
    )[["conversation", "timestamp", "text_clean", "role", "uuid"]]

    # Call the function to test
    result_df = concatenate_consecutive_roles(df)
    print(result_df)

    # Assert that the result matches the expected dataframe
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_process_bot_qs(mock_dataframe):
    expected_bot_qs_list = ["How are you?", "Good night"]
    bot_qs_list, bot_qs_df = process_bot_qs(mock_dataframe)
    assert bot_qs_list == expected_bot_qs_list, "Bot question list should match expected sentences"
    # Check that the sentences are exploded correctly
    assert bot_qs_df["sentences"].tolist() == expected_bot_qs_list, "Exploded sentences should match expected values"


@patch("dsp_interview_transcripts.pipeline.process_data.SENTENCE_MODEL.encode")
def test_match_questions(mock_encode, mock_questions):
    # Simulate encoding outputs
    mock_encode.side_effect = lambda x: [[1, 0], [0, 1], [0.5, 0.5]] if x == mock_questions else [[1, 0], [0.5, 0.5]]
    bot_qs_list = ["How are you?", "Good night"]

    expected_matches = pd.DataFrame(
        {
            "bot_q": ["How are you?", "Good night"],
            "question": ["How are you?", "Good night?"],
            "cosine_similarity": [1.0, 1.0],
        }
    )

    final_matches = match_questions(bot_qs_list, mock_questions, threshold=0.8)
    print(final_matches)
    pd.testing.assert_frame_equal(final_matches.reset_index(drop=True), expected_matches)


def test_get_best_matches(mock_dataframe, mock_questions):
    # Mock the results of previous steps
    bot_qs_list, bot_qs_df = process_bot_qs(mock_dataframe)
    final_matches = pd.DataFrame(
        {
            "bot_q": ["How are you?", "Good night"],
            "question": ["How are you?", "Good night?"],
            "cosine_similarity": [1.0, 1.0],
        }
    )
    questions_df = pd.DataFrame(enumerate(mock_questions), columns=["q_number", "question"])

    # Expected output DataFrame
    expected_data = {
        "conversation": [1, 2],
        "text_clean": ["How are you?", "Good night"],
        "role": ["BOT", "BOT"],
        "sentences": ["How are you?", "Good night"],
        "question": ["How are you?", "Good night?"],
        "cosine_similarity": [1.0, 1.0],
    }
    expected_df = pd.DataFrame(expected_data)

    best_matches = get_best_matches(bot_qs_df, final_matches, questions_df)
    pd.testing.assert_frame_equal(best_matches[expected_df.columns], expected_df)
