from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts.getters.data_getters import load_s3_data


def get_full_output(production: bool = False, config=config):
    """Get the full output data

    Args:
        production (bool, optional): Do you want production data? Defaults to False.
        config: Config containing production and test paths. Defaults to config.

    Returns:
        pd.DataFrame with the following columns:
        Columns:
        - Topic Name: The name of the topic associated with a user response, or 'None' if it's the noise cluster.
        - Topic Description: Brief description of the topic.
        - Topic Top Words: TFIDF representation of the topic.
        - Representative_of_topic: A binary indicator (1 or 0) denoting whether the user response in this row is representative of the topic.
        - probable question: The most likely question that a user's response corresponds to (see process_data.py).
        - context: The preceding BOT and USER responses before the current one.
        - user_response: The current user response on which topic assignment etc is based.
        - predicted_sentiment: The predicted sentiment of the user response.
        - conversation: A unique identifier for the conversation.
        - uuid: A unique identifier for the user response.
        - timestamp: Timestamp of the user response.
    """
    if production:
        DATA_PATH = config["prod_paths"]["final_full_data_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["final_full_data_s3_path"]

    return load_s3_data(S3_BUCKET, DATA_PATH)


def get_summary_table(production: bool = False, config=config):
    """Get the distilled output ie just the topic names and descriptions,
    and the representative documents (user responses) from each topic.

    Args:
        production (bool, optional): Do you want production data? Defaults to False.
        config: Config containing production and test paths. Defaults to config.

    Returns:
        pd.DataFrame with the following columns:
        ["Name", "Description", "Top Words", "N responses in topic", "conversation", "uuid","text_clean", "context", "sentiment"]
    """
    if production:
        DATA_PATH = config["prod_paths"]["final_summary_info_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["final_summary_info_s3_path"]

    return load_s3_data(S3_BUCKET, DATA_PATH)
