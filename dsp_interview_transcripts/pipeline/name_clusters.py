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


class NameDescription(BaseModel):
    """Model for naming and describing a group of documents."""

    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


prompt = """
    I have performed text clustering on some interviews where professionals were asked about the occupational risks of sedentary behaviour at
    work, and the challenges organisations face in reducing sedentary behaviour.
    \n
    One of the clusters contains the following user responses from the interviews:
    {docs}
    The cluster is described by the following keywords: {keywords}
    \n
    Based on the information above, please provide a **French language** name and description for the cluster as a JSON object with two fields:
    - name: A short, informative name for the cluster **in French**
    - description: A short description of the cluster, based on the user responses and keywords provided **in French**
    \n
    Provide nothing except for this JSON dict.
    \n
    """

parser = JsonOutputParser(pydantic_object=NameDescription)

final_prompt = PromptTemplate(
    template=prompt,
    input_variables=["docs", "keywords"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = "llama3.2"

ollama_model = ChatOllama(model=model, temperature=0)

llm_chain = final_prompt | ollama_model | parser


def name_topics(
    topic_info: pd.DataFrame,
    llm_chain,
    text_col: str = "text_clean",
    top_words_col: str = "Representation",
    topic_label_col: str = "Topic",
) -> Dict[str, dict]:
    """
    Generate names and descriptions for each topic by invoking an LLM chain,
    using representative text and keywords for each topic.

    Args:
        topic_info (pd.DataFrame): A DataFrame containing topic information,
            including columns for text samples, top words, and topic labels.
        llm_chain: A language model chain used for generating topic names and descriptions.
            It must support an `invoke` method that accepts a dictionary with 'docs' and 'keywords' keys.
        topics (List[str]): A list of topic identifiers to process.
        text_col (str, optional): Column name in `topic_info` containing the text data for each topic.
            Defaults to 'text_clean'.
        top_words_col (str, optional): Column name in `topic_info` containing the top words for each topic.
            Defaults to 'Top Words'.
        topic_label_col (str, optional): Column name in `topic_info` that indicates topic labels.
            Defaults to 'Cluster'.

    Returns:
        Dict[str, dict]: A dictionary where each key is a topic identifier and each value is
        a dictionary with the generated 'name' and 'description' for that topic.

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
        logger.info(f"Docs: {docs}")
        keywords = temp_df[top_words_col].values[0]
        logger.info(f"Keywords: {keywords}")

        try:
            output = llm_chain.invoke({"docs": docs, "keywords": keywords})
            logger.info(f"Generated name: {output['name']}, description: {output['description']}")
            results[topic] = output

        except Exception as e:
            logger.error(f"Error processing topic {topic}: {str(e)}")
            results[topic] = {"error": str(e)}

    return results


def main(production: bool = False):

    # MIN_LEN = config["min_length"]
    # if production:
    #     OUT_PATH = config["prod_paths"]["interim_w_names_s3_path"].format(MIN_LEN=MIN_LEN)
    # else:
    #     OUT_PATH = config["test_paths"]["interim_w_names_s3_path"].format(MIN_LEN=MIN_LEN)

    # topic_info = get_rep_docs(production=production)

    topic_info = pd.read_csv(f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/repr_docs.csv")

    topic_info = topic_info.groupby(["Topic", "Name", "Representation"])["text"].apply(list).reset_index()
    topic_info["Topic"] = topic_info["Topic"].astype(str)

    results = name_topics(
        topic_info, llm_chain, text_col="text", top_words_col="Representation", topic_label_col="Topic"
    )

    # Some complicated conditionals to check that what's in `results` can be parsed
    topic_info[f"{model}_name"] = topic_info["Topic"].map(
        lambda x: results[x]["name"]
        if x in results and isinstance(results[x], dict) and "name" in results[x]
        else None
    )
    topic_info[f"{model}_description"] = topic_info["Topic"].map(
        lambda x: results[x]["description"]
        if x in results and isinstance(results[x], dict) and "description" in results[x]
        else None
    )

    logger.info("Saving output...")
    # save_to_s3(S3_BUCKET, topic_info, OUT_PATH)
    topic_info.to_csv(
        f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/topic_names_and_descriptions.csv"
    )
    logger.info("Done!")


if __name__ == "__main__":
    plac.call(main)
