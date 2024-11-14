"""Use a llama model to give names and descriptions for the topics."""
from typing import Dict

import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from pydantic import Field

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


class NameDescription(BaseModel):
    """Model for naming and describing a group of documents."""

    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")


prompt = """
    I have performed text clustering on some interviews where users were asked about their knowledge of
    and opinions on different home heating options. In the interview, users were asked about their knowledge
    of the Boiler Upgrade Scheme, a scheme that provides a subsidy to homeowners wishing to install a heatpump
    instead of getting a new gas boiler for their home.
    \n
    One of the clusters contains the following user responses from the interviews:
    {docs}
    The cluster is described by the following keywords: {keywords}
    \n
    Based on the information above, please provide a name and summary for the cluster as a JSON object with two fields:
    - name: A short, informative name for the cluster
    - description: A summary of views of users within the cluster. You can include sentiments they express, reasons for their views, their knowledge levels, and any other relevant information.
    \n
    Provide nothing except for this JSON dict.
    \n
    Example:
    {{
        "name": "Energy Efficiency",
        "description": "This cluster contains user responses about energy efficiency when choosing home heating options. The users have varying degrees of knowledge about the efficiency of different systems. Some reasons for wanting to improve efficiency include environmental concerns and cost concerns."
    }}
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

INPUT_PATH = PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv"


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


if __name__ == "__main__":

    topic_info = pd.read_csv(INPUT_PATH)

    topic_info = topic_info.groupby(["Topic", "Representation"])["text_clean"].apply(list).reset_index()
    topic_info["Topic"] = topic_info["Topic"].astype(str)

    results = name_topics(
        topic_info, llm_chain, text_col="text_clean", top_words_col="Representation", topic_label_col="Topic"
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
    topic_info.to_csv(
        PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv", index=False
    )
    logger.info("Done!")
