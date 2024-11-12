""" Getter for extracting interim data: data with topics added; representative docs; the names and descriptions for clusters. """
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts.getters.data_getters import load_s3_data


MIN_LEN = config["min_length"]


def get_cleaned_data(production=False, config=config, min_len=MIN_LEN):
    """Data produced by `process_data.py"""
    if production:
        DATA_PATH = config["prod_paths"]["interim_processed_data_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["interim_processed_data_s3_path"]
    DATA_PATH = DATA_PATH.format(MIN_LEN=min_len)
    return load_s3_data(S3_BUCKET, DATA_PATH)


def get_data_w_topics(production=False, config=config, min_len=MIN_LEN):

    if production:
        DATA_PATH = config["prod_paths"]["interim_data_w_topics_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["interim_data_w_topics_s3_path"]
    DATA_PATH = DATA_PATH.format(MIN_LEN=min_len)
    return load_s3_data(S3_BUCKET, DATA_PATH)


def get_rep_docs(production=False, config=config, min_len=MIN_LEN):

    if production:
        DATA_PATH = config["prod_paths"]["interim_representative_docs_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["interim_representative_docs_s3_path"]

    DATA_PATH = DATA_PATH.format(MIN_LEN=min_len)
    return load_s3_data(S3_BUCKET, DATA_PATH)


def get_topic_names(production=False, config=config, min_len=MIN_LEN):

    if production:
        DATA_PATH = config["prod_paths"]["interim_w_names_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["interim_w_names_s3_path"]

    DATA_PATH = DATA_PATH.format(MIN_LEN=min_len)
    return load_s3_data(S3_BUCKET, DATA_PATH)
