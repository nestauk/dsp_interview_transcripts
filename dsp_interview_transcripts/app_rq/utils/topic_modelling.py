"""
Functions that do the heavy lifting of the topic modelling page of the app.
Later these could be moved to FastAPI?
"""
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union
from typing import Optional

import os
import pandas as pd

from umap import UMAP

from pydantic import BaseModel
from pydantic import Field
import logging

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable

import numpy as np

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# from dsp_interview_transcripts import PROJECT_DIR
# from dsp_interview_transcripts.utils.llm_utils import NameDescription
# from dsp_interview_transcripts.utils.llm_utils import format_output_df
# from dsp_interview_transcripts.utils.llm_utils import get_chain
# from dsp_interview_transcripts.utils.llm_utils import name_topics
# from dsp_interview_transcripts.utils.topic_modelling import embed_docs
# from dsp_interview_transcripts.utils.topic_modelling import init_topic_model


MODEL = "llama3.2"
# base_dir = Path(__file__).parent
BASIC_PROMPT_PATH = (Path(__file__).parent / "../prompts/basic_prompt.txt").resolve()

def embed_docs(
    docs: List[str], model: Optional[SentenceTransformer] = None, save: bool = False, outpath: str = "embeddings.npy"
) -> Tuple[List[str], np.ndarray]:
    """Embeds a list of documents using a SentenceTransformer model.
    Saves these to the local path specified as `outpath`.

    Args:
        docs (List[str]): List of text documents
        model (Optional[SentenceTransformer], optional): SentenceTransformer model to use. Defaults to None.
        save (bool, optional): Do you want the embeddings saved as `npy`? Defaults to False.
        outpath (str, optional): Local path for saving the embeddings - only used if `save==True`. Defaults to "embeddings.npy".

    Returns:
        Tuple[List[str], np.ndarray]: The input documents and their embeddings.
    """
    logging.info("Embedding user messages...")

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(docs, show_progress_bar=True)
    if save:
        np.save(outpath, embeddings)
    return docs, embeddings


def init_topic_model(
    stop_words: Union[str, List[str]],
    min_cluster_size: int,
    hdbscan_selection_method: str,
    embedding_model: SentenceTransformer,
    seed: int = 42,
    empty_reduction: bool = False,
    nr_topics=None,
) -> Tuple[BERTopic, TfidfVectorizer, Dict[str, Union[KeyBERTInspired, MaximalMarginalRelevance]]]:
    """Initializes and returns a BERTopic model along with vectorizer and representation models.
    The representation model and vectorizer can be reused later for noise reduction.

    Args:
        stop_words (Union[str, List[str]]): Stopwords to use for the vectorizer
        min_cluster_size (int): The smallest size of a cluster with HDBSCAN
        hdbscan_selection_method (str): "eom" or "leaf"
        embedding_model (SentenceTransformer): SentenceTransformer model to use for embeddings
        seed (int, optional): Random seed. Defaults to 42.
        empty_reduction (bool, optional): You can specify an empty reduction model if you have already reduced the embeddings. Defaults to False and allowing BERTopic to do the reduction.

    Returns:
        Tuple[BERTopic, TfidfVectorizer, Dict[str, Union[KeyBERTInspired, MaximalMarginalRelevance]]]:
            BERTopic model, vectorizer model, and representation models
    """

    if empty_reduction:
        reduction_model = BaseDimensionalityReduction()
    else:
        reduction_model = UMAP(
            n_neighbors=15,
            n_components=50,
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )

    hdbscan_model = HDBSCAN(
        min_samples=5,
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method=hdbscan_selection_method,
        prediction_data=True,
    )

    vectorizer_model = TfidfVectorizer(
        stop_words=stop_words,
        min_df=1,
        max_df=0.85,
        ngram_range=(1, 3),
    )

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # All representation models
    representation_model = {
        "KeyBERT": keybert_model,
        "MMR": mmr_model,
    }

    if nr_topics is not None:
        topic_model = BERTopic(
            # Pipeline models
            embedding_model=embedding_model,
            umap_model=reduction_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            nr_topics=nr_topics,
            # Hyperparameters
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
        )
    else:
        topic_model = BERTopic(
            # Pipeline models
            embedding_model=embedding_model,
            umap_model=reduction_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            # Hyperparameters
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
        )

    return topic_model, vectorizer_model, representation_model


class NameDescription(BaseModel):
    """Model for naming and describing a group of documents."""

    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")

def load_prompt_template(prompt_path: Path) -> str:
    """Load the prompt template from a file."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def get_llm(provider, model, temp):

    if provider == "ollama":
        llm = ChatOllama(model=model, temperature=temp)
    elif provider == "azure":
        llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=temp,
        )
    return llm

def get_chain(
    prompt_path: Union[Path, str],
    input_vars: List[str],
    output_template: Type[BaseModel],
    provider="ollama",
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

    llm = get_llm(provider, model, temp)

    llm_chain = final_prompt | llm | parser

    return llm_chain

LLM_CHAIN = get_chain(
    BASIC_PROMPT_PATH,
    input_vars=["docs", "keywords"],
    output_template=NameDescription,
    provider="ollama",
    model=MODEL,
    temp=0,
)


def get_topics_and_summaries(
    user_messages: pd.DataFrame,
    text_col: str,
    num_topics: int = 10,
    model=MODEL,
    llm_chain=LLM_CHAIN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs topic modelling and summarization on the dataset uploaded by the user,
    with the maximum number of topics defined in the input in the topic modelling page of the app.

    The dataframe `user_messages` is so called because there is a callback in the app that filters the original
    dataframe to just rows where `role=='USER'`. This is not ideal long term.

    This function:
    - embeds the documents
    - fits a BERTopic model to identify topics,
    - reduces the noise cluster
    - uses llama3.2 to generate names and descriptions for the topics
    - creates a datafrme ready for visualisation, with 2D UMAP projections of the embeddings

    Args:
        user_messages (pd.DataFrame): DataFrame containing user text data.
        text_col (str): Name of the column in `user_messages` that contains the text data.
        num_topics (int, optional): Desired number of topics for the model to extract. Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - `df_vis`: DataFrame containing 2D UMAP projections, topic assignments, topic names, and merged original data. One row per user response.
            - `topic_lookup`: DataFrame containing topic metadata, including topic number, name, and LLM-generated descriptions. One row per topic.

    Notes:
        - Uses BERTopic with HDBSCAN and UMAP.
        - Assumes global variables `llm_chain` and `MODEL` are available for naming topics with LLM.
        - Assumes helper functions: `embed_docs`, `init_topic_model`, `name_topics`, and `format_output_df`.
    """
    # Drop rows where the text column is not a string or is missing
    user_messages = user_messages[user_messages[text_col].apply(lambda x: isinstance(x, str))]

    # Convert all values to string just to be extra safe (in case of mixed types)
    user_messages[text_col] = user_messages[text_col].astype(str)

    docs = user_messages[text_col].tolist()

    docs, embeddings = embed_docs(docs, save=False)

    topic_model, vectorizer_model, representation_model = init_topic_model(
        stop_words="english",
        min_cluster_size=10,
        hdbscan_selection_method="leaf",
        embedding_model="all-MiniLM-L6-v2",
        seed=42,
        empty_reduction=False,
        nr_topics=num_topics,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings)

    new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")

    topic_model.update_topics(
        docs,
        topics=new_topics,
        top_n_words=10,
        n_gram_range=(1, 3),
        vectorizer_model=vectorizer_model,
        ctfidf_model=None,
        representation_model=representation_model,
    )

    summary_info = topic_model.get_topic_info()

    results = name_topics(
        summary_info,
        llm_chain,
        input_variable_dict={"docs": "Representative_Docs", "keywords": "Representation"},
        topic_label_col="Topic",
    )

    topic_info = format_output_df(
        topic_info=summary_info, results=results, output_fields=["name", "description"], model_name=model
    )

    umap_2d = UMAP(random_state=42, n_components=2)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    topic_lookup = topic_info[["Topic", "Name", "Representation", f"{model}_name", f"{model}_description"]]

    df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_vis["topic"] = new_topics
    df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
    df_vis["doc"] = docs

    df_vis = pd.merge(
        user_messages,
        df_vis,
        left_on=text_col,
        right_on="doc",
        how="outer",
    )

    return df_vis, topic_lookup


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
        logging.info(f"Processing topic {topic}")
        temp_df = topic_info[topic_info[topic_label_col] == topic]

        input_data = {
            var_name: temp_df[col_name].values[0] if not temp_df.empty else ""
            for var_name, col_name in input_variable_dict.items()
        }

        try:
            output = llm_chain.invoke(input_data)
            logging.info(output.keys())
            results[topic] = output

        except Exception as e:
            logging.error(f"Error processing topic {topic}: {str(e)}")
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
