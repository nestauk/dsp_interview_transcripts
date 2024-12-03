""" Getter for extracting the unmodified data """
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts.getters.data_getters import load_s3_data


def get_raw_transcripts():
    """Get the unmodified transcripts file"""
    return load_s3_data(S3_BUCKET, "raw/qual_af_transcripts.csv")


def get_raw_transcripts_cleaned(production=False, config=config):
    """Return the transcripts but with some minimal cleaning"""
    if production:
        DATA_PATH = config["prod_paths"]["raw_cleaned_s3_path"]
    else:
        DATA_PATH = config["test_paths"]["raw_cleaned_s3_path"]
    return load_s3_data(S3_BUCKET, DATA_PATH)
