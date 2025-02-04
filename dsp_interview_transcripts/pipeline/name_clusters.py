"""Use a llama model to give names and descriptions for the topics."""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Type

import pandas as pd
import plac

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from pydantic import Field

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


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


def load_prompt_template(prompt_path: Path) -> str:
    """Load the prompt template from a file."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def get_chain(
    prompt_path: Path = PROMPT_PATH,
    input_vars: List[str] = ["docs", "keywords"],
    output_template: Type[BaseModel] = NameDescription,
    model: str = MODEL_NAME,
    temp: float = TEMPERATURE,
):
    """
    Constructs a LangChain processing chain using a prompt template, a language model,
    and a JSON output parser.

    Args:
        prompt_path (Path, optional): Path to text file containing prompt template. Defaults to PROMPT_PATH.
        input_vars (List[str], optional): List of variables expected to be formatted into the prompt. Defaults to ["docs", "keywords"].
        output_template (Type[BaseModel], optional): Pydantic model for the output. Defaults to NameDescription.
        model (str, optional): Name of the language model to use. Defaults to MODEL_NAME.
        temp (float, optional): Temperature setting for the model. Defaults to TEMPERATURE.

    Returns:
        Runnable: A langchain chain
    """

    prompt = load_prompt_template(prompt_path)

    parser = JsonOutputParser(pydantic_object=output_template)

    final_prompt = PromptTemplate(
        template=prompt,
        input_variables=input_vars,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    ollama_model = ChatOllama(model=model, temperature=temp)

    llm_chain = final_prompt | ollama_model | parser

    return llm_chain


def name_topics(
    topic_info: pd.DataFrame,
    llm_chain: Runnable,
    text_col: str = "text_clean",
    top_words_col: str = "Representation",
    topic_label_col: str = "Topic",
) -> Dict[str, Dict[str, Any]]:
    """
    Run an LLM chain over each topic.

    Args:
        topic_info (pd.DataFrame): A DataFrame containing topic information,
            including columns for text samples, top words, and topic labels:
            - `{text_col}`: Column with text samples for each topic.
            - `{top_words_col}`: Column with representative words for each topic.
            - `{topic_label_col}`: Column containing topic identifiers.
        llm_chain: A language model chain used for generating topic names and descriptions.
            It must support an `invoke` method that accepts a dictionary with 'docs' and 'keywords' keys.
        text_col (str, optional): Column name in `topic_info` containing the text data for each topic.
            Defaults to 'text_clean'.
        top_words_col (str, optional): Column name in `topic_info` containing the top words for each topic.
            Defaults to 'Top Words'.
        topic_label_col (str, optional): Column name in `topic_info` that indicates topic labels.
            Defaults to 'Cluster'.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary where each key is a topic identifier (str),
        and each value is a dictionary with the output for that topic. The form of the output
        is determined in the definition of llm_chain.

    Raises:
        Exception: Logs and continues on any exceptions encountered while processing topics,
        capturing errors with the topic identifier and error message.

    Example:
        >>> name_topics(topic_info=df, llm_chain=my_llm_chain, topics=["Topic 1", "Topic 2"])
        {
            "Topic 1": {"name": "Customer Satisfaction", "description": "Documents discussing customer feedback and satisfaction."},
            "Topic 2": {"name": "Product Quality", "description": "Documents focusing on product durability and performance."}
        }
    """
    topics = topic_info[topic_label_col].unique().tolist()

    results = {}

    for topic in topics:
        logger.info(f"Processing topic {topic}")
        temp_df = topic_info[topic_info[topic_label_col] == topic]
        docs = temp_df[text_col].values[0]
        keywords = temp_df[top_words_col].values[0]
        logger.info(f"Keywords: {keywords}")

        try:
            output = llm_chain.invoke({"docs": docs, "keywords": keywords})
            logger.info(output.keys())
            results[topic] = output

        except Exception as e:
            logger.error(f"Error processing topic {topic}: {str(e)}")
            results[topic] = {"error": str(e)}

    return results


def main():

    professions = ["Décideurs", "Salariés", "Elus"]

    llm_chain = get_chain()

    for profession in professions:
        logger.info(f"Processing the interviews of the {profession} group...")
        OUTPATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/{profession}/"

        topic_info = pd.read_csv(f"{OUTPATH}repr_docs.csv")

        topic_info = (
            topic_info.groupby(["Topic", "Name", "Representation"])["context_formatted"].apply(list).reset_index()
        )
        topic_info["Topic"] = topic_info["Topic"].astype(str)

        results = name_topics(
            topic_info,
            llm_chain,
            text_col="context_formatted",
            top_words_col="Representation",
            topic_label_col="Topic",
        )

        # Some complicated conditionals to check that what's in `results` can be parsed:
        for output_group in [
            ("name",),
            ("résumé", "summary", "synthèse"),
            ("maquettes",),
            ("positifs", "positives"),
            ("négatifs", "negatives"),
            ("autres", "other"),
        ]:
            consolidated_output = "_".join(output_group)  # Create a descriptive column name
            topic_info[f"{MODEL_NAME}_{consolidated_output}"] = topic_info["Topic"].map(
                lambda x: next(
                    (
                        results[x][output]
                        for output in output_group
                        if x in results and isinstance(results[x], dict) and output in results[x]
                    ),
                    None,
                )
            )

        logger.info("Saving output...")
        topic_info.to_csv(f"{OUTPATH}topic_names_and_descriptions.csv")
        logger.info("Done!")


if __name__ == "__main__":
    plac.call(main)
