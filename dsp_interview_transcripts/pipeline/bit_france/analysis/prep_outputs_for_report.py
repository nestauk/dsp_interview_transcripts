"""
This script comes after prep_outputs.py :)
"""
import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


REPORT_PATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/report/"

if __name__ == "__main__":

    professions = ["Décideurs", "Salariés", "Elus"]

    for profession in professions:
        logger.info(f"Processing the interviews of the {profession} group...")
        OUTPATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/{profession}/"

        speaker_data_topics = pd.read_csv(f"{OUTPATH}speaker_data_topics.csv")
        speaker_data_topics = speaker_data_topics.rename(
            columns={
                "context_formatted": "Interview excerpt",
            }
        )

        names_descriptions = pd.read_csv(f"{OUTPATH}topic_names_and_descriptions.csv")
        names_descriptions = names_descriptions[
            [
                "Topic",
                "Name",
                "Representation",
                "context_formatted",
                "llama3.2_name",
                "llama3.2_résumé_summary_synthèse",
                "llama3.2_maquettes",
                "llama3.2_positifs_positives",
                "llama3.2_négatifs_negatives",
                "llama3.2_autres_other",
            ]
        ].rename(
            columns={
                "context_formatted": "Interview excerpt",
                "Representation": "Keywords",
                "llama3.2_name": "AI-generated name",
                "llama3.2_résumé_summary_synthèse": "AI-generated summary",
                "llama3.2_maquettes": "AI-identified maquettes",
                "llama3.2_positifs_positives": "AI-identified positives",
                "llama3.2_négatifs_negatives": "AI-identified negatives",
                "llama3.2_autres_other": "AI-identified other info",
            }
        )

        repr_docs = pd.read_csv(f"{OUTPATH}repr_docs.csv")
        repr_docs = repr_docs.rename(columns={"Representation": "Keywords", "context_formatted": "Interview excerpt"})

        repr_docs_final = pd.merge(
            repr_docs,
            names_descriptions[
                [
                    "Name",
                    "AI-generated name",
                    "AI-generated summary",
                    "AI-identified maquettes",
                    "AI-identified positives",
                    "AI-identified negatives",
                    "AI-identified other info",
                ]
            ],
            # names_descriptions[["Name", "llama3.2_name", "llama3.2_description"]],
            on="Name",
            how="left",
        )

        repr_docs_final[
            [
                "Topic",
                "Name",
                "Keywords",
                "AI-generated name",
                "AI-generated summary",
                "AI-identified maquettes",
                "AI-identified positives",
                "AI-identified negatives",
                "AI-identified other info",
                "file_name",
                "Interview excerpt",
            ]
        ].to_csv(f"{REPORT_PATH}{profession}_repr_docs_final.csv", index=False)

        repr_docs_brief = repr_docs_final[
            [
                "Name",
                "Keywords",
                "AI-generated name",
                "AI-generated summary",
                "AI-identified maquettes",
                "AI-identified positives",
                "AI-identified negatives",
                "AI-identified other info",
            ]
        ].drop_duplicates()
        repr_docs_brief.to_csv(f"{REPORT_PATH}{profession}_repr_docs_brief.csv", index=False)

        speaker_data_names_descriptions = pd.merge(speaker_data_topics, repr_docs_brief, on="Name", how="left")
        raw_transcripts = pd.read_csv(f"{PROJECT_DIR}/data/bit_france/converted/combined_transcripts.csv")

        transcripts = pd.merge(
            raw_transcripts[raw_transcripts["profession"] == profession],
            speaker_data_names_descriptions[
                [
                    "text",
                    "Interview excerpt",
                    "file_name",
                    "Name",
                    "Keywords",
                    "AI-generated name",
                    "AI-generated summary",
                    "AI-identified maquettes",
                    "AI-identified positives",
                    "AI-identified negatives",
                    "AI-identified other info",
                ]
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
                "Interview excerpt",
                "Name",
                "Keywords",
                "AI-generated name",
                "AI-generated summary",
                "AI-identified maquettes",
                "AI-identified positives",
                "AI-identified negatives",
                "AI-identified other info",
            ]
        ].to_csv(f"{REPORT_PATH}{profession}_full_data.csv", index=False)
