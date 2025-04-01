"""Use a llama model to give names and descriptions for the topics."""
from typing import Dict

import pandas as pd
import plac

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from pydantic import Field

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts import logger
from dsp_interview_transcripts.getters.data_getters import save_to_s3
from dsp_interview_transcripts.getters.interim import get_rep_docs
from dsp_interview_transcripts.utils.llama_utils import format_output_df
from dsp_interview_transcripts.utils.llama_utils import get_chain
from dsp_interview_transcripts.utils.llama_utils import name_topics


MODEL_NAME = "llama3.2"
TEMPERATURE = 0
PROMPT_PATH = PROJECT_DIR / f"dsp_interview_transcripts/pipeline/prompts/{config['prompt_path']}"


class NameDescription(BaseModel):
    """Model for naming and describing a group of documents."""

    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


@plac.annotations(production=("Run in production mode if True, otherwise in test mode", "flag", "production"))
def main(production: bool = False):

    MIN_LEN = config["min_length"]
    PROJECT = config["project"]

    if production:
        OUT_PATH = f"{PROJECT}/" + config["prod_paths"]["interim_w_names_s3_path"].format(MIN_LEN=MIN_LEN)
    else:
        OUT_PATH = f"{PROJECT}/" + config["test_paths"]["interim_w_names_s3_path"].format(MIN_LEN=MIN_LEN)

    llm_chain = get_chain(
        prompt_path=PROMPT_PATH,
        input_vars=["docs", "keywords"],
        output_template=NameDescription,
        model=MODEL_NAME,
        temp=TEMPERATURE,
    )

    topic_info = get_rep_docs(production=production)

    topic_info = topic_info.groupby(["Topic", "Name", "Representation"])["text_clean"].apply(list).reset_index()
    topic_info["Topic"] = topic_info["Topic"].astype(str)

    results = name_topics(
        topic_info,
        llm_chain,
        input_variable_dict={"docs": "text_clean", "keywords": "Representation"},
        topic_label_col="Topic",
    )

    topic_info = format_output_df(
        topic_info=topic_info, results=results, output_fields=["name", "description"], model_name=MODEL_NAME
    )

    logger.info("Saving output...")
    save_to_s3(S3_BUCKET, topic_info, OUT_PATH)
    logger.info("Done!")


if __name__ == "__main__":
    plac.call(main)
