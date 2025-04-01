from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from pydantic import Field

from dsp_interview_transcripts import logger


def load_prompt_template(prompt_path: Path) -> str:
    """Load the prompt template from a file."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def get_chain(
    prompt_path: Union[Path, str],
    input_vars: List[str],
    output_template: Type[BaseModel],
    model: str = "llama3.2",
    temp: float = 0,
):
    """
    Constructs a LangChain processing chain using a prompt template, a language model,
    and a JSON output parser.

    Args:
        prompt_path (Path): Path to text file containing prompt template.
        input_vars (List[str]): List of variables expected to be formatted into the prompt. Example: ["docs", "keywords"].
        output_template (Type[BaseModel], optional): Pydantic model for the output.
        model (str, optional): Name of the language model to use. Defaults to "llama3.2".
        temp (float, optional): Temperature setting for the model. Defaults to 0.

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
    input_variable_dict: Dict[str, str] = {"docs": "text_clean", "keywords": "Representation"},
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
        input_variable_dict (Dict[str, str]): A dictionary mapping input variable names expected by
            the prompt template to corresponding dataframe column names.
            Example: `{"docs": "text_clean", "keywords": "Representation"}`
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

        input_data = {
            var_name: temp_df[col_name].values[0] if not temp_df.empty else ""
            for var_name, col_name in input_variable_dict.items()
        }

        try:
            output = llm_chain.invoke(input_data)
            logger.info(output.keys())
            results[topic] = output

        except Exception as e:
            logger.error(f"Error processing topic {topic}: {str(e)}")
            results[topic] = {"error": str(e)}

    return results


def format_output_df(
    output_fields: Union[List[str], List[Tuple[str, ...]]],
    topic_info: pd.DataFrame,
    results: Dict[str, Dict[str, str]],
    model_name: str,
) -> pd.DataFrame:
    """Adds as many columns to the output dataframe as you have requested from the LLM
    - for example, if you have just requested "name" and "description" fields,
    you will get back a dataframe with columns "<model-name>_name" and "<model-name>_description".

    `output_fields` can be a list of strings or tuples of strings, where each string is a field name.
    The option for tuples is because of multilingual cases where the model may not be reliable
    about which language it returns the field in. If an output field name is provided as a tuple, the values are concatenated
    using underscores to form the column name.

    If a topic does not exist in `results`, or if an output field is missing for a topic,
    the corresponding entry in the new column will be None.

    Args:
        output_fields : Union[List[str], List[Tuple[str, ...]]]
        A list of output field names, either as strings or tuples of strings.
    topic_info : pd.DataFrame
        A DataFrame containing a "Topic" column.
    results : Dict[str, Dict[str, str]]
        A dictionary where keys represent topic identifiers, and values are dictionaries
        mapping output field names to corresponding values.
    model_name : str
        A prefix to be used when naming new columns in `topic_info` e.g. "llama3.2".

    Returns:
        pd.DataFrame
        The updated `topic_info` DataFrame with new columns named based on `model_name`
        and `output_fields`, containing mapped values from `results`.
    """
    for output_group in output_fields:

        if isinstance(output_group, str):
            output_group = (output_group,)

        consolidated_output = "_".join(output_group)  # Create a descriptive column name
        topic_info[f"{model_name}_{consolidated_output}"] = topic_info["Topic"].map(
            lambda x: next(
                (
                    results[x][output]
                    for output in output_group
                    if x in results and isinstance(results[x], dict) and output in results[x]
                ),
                None,
            )
        )

    return topic_info
