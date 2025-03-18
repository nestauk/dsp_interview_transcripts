"""Use a llama model to give names and descriptions for the topics."""

import pandas as pd
import plac

from pydantic import BaseModel
from pydantic import Field

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger
from dsp_interview_transcripts.utils.llama_utils import format_output_df
from dsp_interview_transcripts.utils.llama_utils import get_chain
from dsp_interview_transcripts.utils.llama_utils import name_topics


MODEL_NAME = "llama3.2"
TEMPERATURE = 0
PROMPT_PATH = PROJECT_DIR / "dsp_interview_transcripts/pipeline/prompts/bit_france_prompt.txt"


class NameDescription(BaseModel):
    """Model for naming and describing a group of documents."""

    name: str = Field(description="Informative name for this group of documents")
    maquettes: str = Field(description="Maquettes mentioned in the texts")
    positives: str = Field(description="Positive attributes of the maquettes")
    negatives: str = Field(description="Negative attributes of the maquettes")
    other: str = Field(description="Other important information")
    summary: str = Field(description="Summary of this group of documents")


def main():

    professions = ["Décideurs", "Salariés", "Elus"]

    for profession in professions:
        logger.info(f"Processing the interviews of the {profession} group...")
        OUTPATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/{profession}/"

        topic_info = pd.read_csv(f"{OUTPATH}repr_docs.csv")

        topic_info = (
            topic_info.groupby(["Topic", "Name", "Representation"])["context_formatted"].apply(list).reset_index()
        )
        topic_info["Topic"] = topic_info["Topic"].astype(str)

        llm_chain = get_chain(
            prompt_path=PROMPT_PATH,
            input_vars=["docs", "keywords"],
            output_template=NameDescription,
            model=MODEL_NAME,
            temp=TEMPERATURE,
        )

        results = name_topics(
            topic_info,
            llm_chain,
            input_variable_dict={"docs": "context_formatted", "keywords": "Representation"},
            topic_label_col="Topic",
        )

        output_fields = [
            ("name",),
            ("résumé", "summary", "synthèse"),
            ("maquettes",),
            ("positifs", "positives"),
            ("négatifs", "negatives"),
            ("autres", "other"),
        ]
        topic_info = format_output_df(
            topic_info=topic_info, results=results, output_fields=output_fields, model_name=MODEL_NAME
        )

        logger.info("Saving output...")
        topic_info.to_csv(f"{OUTPATH}topic_names_and_descriptions.csv")
        logger.info("Done!")


if __name__ == "__main__":
    plac.call(main)
