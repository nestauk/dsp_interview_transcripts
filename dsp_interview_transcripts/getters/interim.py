""" Getter for extracting interim data: data with topics added; representative docs; the names and descriptions for clusters. """
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts.getters.data_getters import load_s3_data


MIN_LEN = config["min_length"]
PROJECT = config["project"]

if PROJECT == "heatflex":
    project = "heatflex/"
else:
    project = ""


def get_data(data_name, production=False, config=config, min_len=MIN_LEN):
    paths_key = "prod_paths" if production else "test_paths"
    s3_path = config[paths_key][f"interim_{data_name}_s3_path"]
    formatted_path = s3_path.format(MIN_LEN=min_len)
    formatted_path = f"{project}{formatted_path}"
    return load_s3_data(S3_BUCKET, formatted_path)


def get_cleaned_data(production=False, config=config, min_len=MIN_LEN):
    """Data produced by `process_data.py"""
    return get_data("processed_data", production, config, min_len)


def get_data_w_topics(production=False, config=config, min_len=MIN_LEN):
    return get_data("data_w_topics", production, config, min_len)


def get_rep_docs(production=False, config=config, min_len=MIN_LEN):
    return get_data("representative_docs", production, config, min_len)


def get_topic_names(production=False, config=config, min_len=MIN_LEN):
    return get_data("w_names", production, config, min_len)
