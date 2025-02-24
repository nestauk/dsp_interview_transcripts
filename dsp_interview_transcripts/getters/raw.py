""" Getter for extracting the unmodified data """
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts.getters.data_getters import load_s3_data


PROJECT = config["project"]
RAW_FILE = config["raw_filepath"]

if PROJECT == "heatflex":
    project = "heatflex/"
else:
    project = ""

RAW_PATH = f"{project}raw/{RAW_FILE}"


def get_raw_transcripts():
    """Get the unmodified transcripts file"""
    return load_s3_data(S3_BUCKET, RAW_PATH)


def get_raw_transcripts_cleaned():
    """Return the transcripts but with some minimal cleaning to:
    - remove NAs
    - merge text answers and audio transcribed answers
    """
    transcripts = load_s3_data(S3_BUCKET, RAW_PATH)
    transcripts = (
        transcripts.assign(text=lambda x: x["text"].fillna(x["transcript"]))
        .fillna({"text": ""})
        .assign(text_length=lambda x: x["text"].apply(len))
    )
    return transcripts
