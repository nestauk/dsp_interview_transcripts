"""
Functions that do the heavy lifting of the topic modelling page of the app.
Later these could be moved to FastAPI?
"""
from typing import Tuple

import pandas as pd

from umap import UMAP

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts.utils.llm_utils import NameDescription
from dsp_interview_transcripts.utils.llm_utils import format_output_df
from dsp_interview_transcripts.utils.llm_utils import get_chain
from dsp_interview_transcripts.utils.llm_utils import name_topics
from dsp_interview_transcripts.utils.topic_modelling import embed_docs
from dsp_interview_transcripts.utils.topic_modelling import init_topic_model


MODEL = "llama3.2"

llm_chain = get_chain(
    PROJECT_DIR / "dsp_interview_transcripts/pipeline/prompts/basic_prompt.txt",
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
    llm_chain=llm_chain,
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
