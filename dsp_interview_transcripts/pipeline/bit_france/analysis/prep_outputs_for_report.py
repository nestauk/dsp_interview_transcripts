"""
This script comes after prep_outputs.py :)
"""
import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR


DATA_PATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france"

if __name__ == "__main__":

    speaker_data_topics = pd.read_csv(f"{DATA_PATH}/outputs/speaker_data_topics.csv")

    names_descriptions = pd.read_csv(f"{DATA_PATH}/outputs/topic_names_and_descriptions.csv")

    repr_docs = pd.read_csv(f"{DATA_PATH}/outputs/repr_docs.csv")

    repr_docs_final = pd.merge(
        repr_docs, names_descriptions[["Name", "llama3.2_name", "llama3.2_description"]], on="Name", how="left"
    )
    repr_docs_final = repr_docs_final[
        ["Topic", "Name", "Representation", "llama3.2_name", "llama3.2_description", "profession", "file_name", "text"]
    ].rename(
        columns={
            "Representation": "Keywords",
            "llama3.2_name": "AI-generated name",
            "llama3.2_description": "AI-generated description",
        }
    )

    repr_docs_final.to_csv(f"{DATA_PATH}/report/repr_docs_final.csv", index=False)

    repr_docs_brief = repr_docs_final[
        ["Topic", "Name", "Keywords", "AI-generated name", "AI-generated description"]
    ].drop_duplicates()
    repr_docs_brief.to_csv(f"{DATA_PATH}/report/repr_docs_brief.csv", index=False)

    speaker_data_names_descriptions = pd.merge(speaker_data_topics, repr_docs_brief, on="Name", how="left")

    raw_transcripts = pd.read_csv(f"{PROJECT_DIR}/data/bit_france/converted/combined_transcripts.csv")

    transcripts = pd.merge(
        raw_transcripts,
        speaker_data_names_descriptions[
            ["text", "file_name", "Name", "Keywords", "AI-generated name", "AI-generated description"]
        ],
        on=["file_name", "text"],
        how="left",
    )

    transcripts[
        [
            "profession",
            "file_name",
            "speaker_id",
            "text",
            "Name",
            "Keywords",
            "AI-generated name",
            "AI-generated description",
        ]
    ].to_csv(f"{DATA_PATH}/report/full_data.csv", index=False)
