"""
Assign interview roles based on mapping table
"""
import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


def assign_role(row):
    # This rule is an exception because for some reason this file name doesn't parse
    if "E25" in row["file_name"] and row["speaker_id"] == 0:
        return "informant"
    if row["informant"] == 0 and row["speaker_id"] == 0:
        return "informant"
    elif row["informant"] == 1 and row["speaker_id"] == 1:
        return "informant"
    else:
        return "other"


if __name__ == "__main__":
    mapping = pd.read_csv(f"{PROJECT_DIR}/data/bit_france/converted/file_mapping.csv")
    data = pd.read_csv(f"{PROJECT_DIR}/data/bit_france/converted/combined_transcripts.csv")

    data_with_mapping = data.merge(mapping[["file_name", "informant"]], on="file_name", how="left")

    data_with_mapping["role"] = data_with_mapping.apply(assign_role, axis=1)
    roles = data_with_mapping[data_with_mapping["role"] == "informant"][["file_name", "speaker_id"]].drop_duplicates()
    for idx, row in roles.iterrows():
        logger.info(f"File: {row['file_name']} Speaker ID: {row['speaker_id']}")

    data_with_mapping.to_csv(
        f"{PROJECT_DIR}/data/bit_france/converted/combined_transcripts_with_roles.csv", index=False
    )
