""" Getter for extracting the unmodified data """
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts.getters.data_getters import load_s3_data


def get_raw_transcripts():
    """Get the unmodified transcripts file"""
    return load_s3_data(S3_BUCKET, "raw/qual_af_transcripts.csv")


def get_raw_transcripts_cleaned():
    """Return the transcripts but with some minimal cleaning to:
    - remove NAs
    - merge text answers and audio transcribed answers
    """
    transcripts = load_s3_data(S3_BUCKET, "raw/qual_af_transcripts.csv")
    transcripts = (
        transcripts.assign(text=lambda x: x["text"].fillna(x["transcript"]))
        .fillna({"text": ""})
        .assign(text_length=lambda x: x["text"].apply(len))
    )
    return transcripts
