import ast
import math
import os
import re

import cluster_analysis_utils
import numpy as np
import pandas as pd
import plotly.express as px

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts.utils.repr_docs import *


lemmatizer = WordNetLemmatizer()

french_stopwords = stopwords.words("french") + ["aussi", "moi"]


def preproc(text: str) -> str:
    text = re.sub(r"[\.,]+", "", text).lower()
    text = cluster_analysis_utils.simple_tokenizer(text)
    text = [t for t in text if t not in french_stopwords]
    return " ".join(text)


def merge_2d_embeddings(speaker_data_topics):
    speaker_data_topics["embeddings"] = speaker_data_topics["embeddings"].apply(ast.literal_eval)
    embeddings_array = np.vstack(speaker_data_topics["embeddings"].values)

    umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=42)
    embeddings_2d = umap_2d.fit_transform(embeddings_array)

    embeddings_df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    return pd.concat([speaker_data_topics, embeddings_df], axis=1)


def save_repr_docs(speaker_data_topics, outpath, text_col="text"):
    _, clustered_data = get_min_radius(
        speaker_data_topics, k_neighbours=10, topic_col="topic", embedding_col="embeddings", metric="cosine"
    )

    clustered_data["quartile"] = clustered_data.groupby("topic")["radius_10"].transform(
        lambda x: pd.qcut(x, q=4, labels=["1st", "2nd", "3rd", "greater than 3rd"])
    )
    repr_docs = extract_repr_docs(clustered_data, n=10, random_seed=42, user_id_col="file_name")

    repr_docs = repr_docs[["Topic", "Name", "Representation", "file_name", "profession", text_col]]

    repr_docs.to_csv(outpath, index=False)

    return repr_docs


def get_data_w_keywords(speaker_data_topics, text_col="text", n_clusters=30):
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    clusterer.fit(speaker_data_topics[["x", "y"]])
    soft_clusters = list(clusterer.labels_)

    texts = speaker_data_topics[text_col].apply(preproc)

    cluster_texts = cluster_analysis_utils.cluster_texts(texts, soft_clusters)

    cluster_keywords = cluster_analysis_utils.cluster_keywords(
        documents=list(cluster_texts.values()),
        cluster_labels=list(cluster_texts.keys()),
        n=2,
        max_df=0.90,
        min_df=0.01,
        Vectorizer=TfidfVectorizer,
    )

    speaker_data_topics["soft_cluster"] = soft_clusters
    speaker_data_topics["soft_cluster_"] = [str(x) for x in soft_clusters]

    centroids = (
        speaker_data_topics.groupby("soft_cluster")
        .agg(x_c=("x", "mean"), y_c=("y", "mean"))
        .reset_index()
        .assign(keywords=lambda x: x.soft_cluster.apply(lambda y: ", ".join(cluster_keywords[y])))
    )

    data_viz = pd.merge(speaker_data_topics, centroids, on="soft_cluster", how="left")

    data_viz.loc[data_viz["Topic"] == -1, "keywords"] = ""

    return data_viz, centroids


def get_topics_by_profession(speaker_data_topics):

    speaker_data_grouped = (
        speaker_data_topics.groupby(["profession", "Topic", "Name"])
        .agg(
            # median_word_count=('word_count', 'median'),
            unique_file_names=("file_name", "nunique"),
            count=("file_name", "count"),
        )
        .reset_index()
        .rename(columns={"profession": "informant"})
    )

    # Calculate the total count of records per profession
    speaker_data_grouped["total_count"] = speaker_data_grouped.groupby("informant")["count"].transform("sum")

    # Calculate the proportion for each topic within each profession
    speaker_data_grouped["proportion"] = speaker_data_grouped["count"] / speaker_data_grouped["total_count"]

    return speaker_data_grouped


if __name__ == "__main__":
    professions = ["Décideurs", "Salariés", "Elus"]

    for profession in professions:
        logger.info(f"Processing the interviews of the {profession} group...")
        OUTPATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/{profession}/"
        os.makedirs(OUTPATH, exist_ok=True)

        speaker_data_topics = pd.read_csv(
            f"{OUTPATH}speaker_data_topics_selection_leaf_min_length_9_min_cluster_15_red_probabilities.csv"
        )
        topic_info = pd.read_csv(
            f"{OUTPATH}bertopic_topic_info_selection_leaf_min_length_9_min_cluster_15_red_probabilities.csv"
        )

        speaker_data_topics = merge_2d_embeddings(speaker_data_topics)

        speaker_data_topics = pd.merge(
            speaker_data_topics,
            topic_info[["Topic", "Name", "Representation"]],
            left_on="topic",
            right_on="Topic",
            how="left",
        )

        speaker_data_topics.to_csv(f"{OUTPATH}speaker_data_topics.csv", index=False)

        repr_docs = save_repr_docs(
            speaker_data_topics, outpath=f"{OUTPATH}repr_docs.csv", text_col="context_formatted"
        )

        n_clusters = len(speaker_data_topics["Topic"].unique()) * 2

        data_viz, centroids = get_data_w_keywords(
            speaker_data_topics, text_col="context_formatted", n_clusters=n_clusters
        )

        data_viz.to_csv(
            f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/report/{profession}_data_viz.csv",
            index=False,
        )
        centroids.to_csv(
            f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/report/{profession}_centroids.csv",
            index=False,
        )
