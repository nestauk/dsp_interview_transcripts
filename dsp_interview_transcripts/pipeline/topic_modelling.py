import random

import numpy as np
import pandas as pd
import plac
import torch

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts import logger
from dsp_interview_transcripts.getters.data_getters import save_to_s3
from dsp_interview_transcripts.getters.interim import get_cleaned_data
from dsp_interview_transcripts.utils.repr_docs import *
from dsp_interview_transcripts.utils.topic_modelling import embed_docs
from dsp_interview_transcripts.utils.topic_modelling import get_proportion_noise
from dsp_interview_transcripts.utils.topic_modelling import init_topic_model


# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

MIN_CLUSTER_SIZE = 20


def main(production: bool = False):

    MIN_LEN = config["min_length"]

    if production:
        OUT_PATH_FULL_DATA = config["prod_paths"]["interim_data_w_topics_s3_path"].format(MIN_LEN=MIN_LEN)
        OUT_PATH_REP_DOCS = config["prod_paths"]["interim_representative_docs_s3_path"].format(MIN_LEN=MIN_LEN)
    else:
        OUT_PATH_FULL_DATA = config["test_paths"]["interim_data_w_topics_s3_path"].format(MIN_LEN=MIN_LEN)
        OUT_PATH_REP_DOCS = config["test_paths"]["interim_representative_docs_s3_path"].format(MIN_LEN=MIN_LEN)

    user_messages = get_cleaned_data(production=production)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=50,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    topic_model, _, _ = init_topic_model(
        stop_words="english",
        min_cluster_size=MIN_CLUSTER_SIZE,
        hdbscan_selection_method="eom",
        embedding_model=model_name,
        seed=RANDOM_SEED,
        empty_reduction=True,
    )

    docs = user_messages["text_clean"].tolist()
    docs, embeddings = embed_docs(docs, SENTENCE_MODEL, save=False)

    embeddings_50d = umap_model.fit_transform(embeddings)

    normalized_embeddings = normalize(embeddings_50d, norm="l2")

    topics, _ = topic_model.fit_transform(docs, normalized_embeddings)

    summary_info = topic_model.get_topic_info()
    if production:
        bertopic_summary_outpath = "interim/bertopic_topic_info.csv"
    else:
        bertopic_summary_outpath = "test/interim/bertopic_topic_info.csv"
    save_to_s3(
        S3_BUCKET,
        summary_info,
        bertopic_summary_outpath,
    )

    umap_2d = UMAP(random_state=RANDOM_SEED, n_components=2)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    topic_lookup = summary_info[["Topic", "Name", "Representation"]]

    df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_vis["topic"] = topics
    df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
    df_vis["doc"] = docs

    df_vis = pd.merge(
        user_messages[["uuid", "conversation", "timestamp", "text_clean", "sentiment", "question", "context"]],
        df_vis,
        left_on="text_clean",
        right_on="doc",
        how="outer",
    )

    logger.info(f'Topic distribution: {df_vis["Name"].value_counts(normalize=True)}')

    unique_conversations_per_topic = df_vis.groupby("Topic")["conversation"].nunique().reset_index()
    unique_conversations_per_topic.columns = ["Topic", "N_users"]
    logger.info(f"N users in each topic: {unique_conversations_per_topic}")

    df_vis["norm_embedding"] = list(normalized_embeddings)

    logger.info("Saving data...")
    save_to_s3(S3_BUCKET, df_vis, OUT_PATH_FULL_DATA)

    # save most representative documents
    df_vis_no_noise = df_vis[df_vis["topic"] != -1]
    _, clustered_data = get_min_radius(df_vis_no_noise, k_neighbours=10)

    clustered_data["quartile"] = clustered_data.groupby("topic")["radius_10"].transform(
        lambda x: pd.qcut(x, q=4, labels=["1st", "2nd", "3rd", "greater than 3rd"])
    )
    repr_docs = extract_repr_docs(clustered_data)

    logger.info("Saving data...")
    save_to_s3(
        S3_BUCKET,
        repr_docs[
            [
                "Topic",
                "Name",
                "Representation",
                "text_clean",
                "sentiment",
                "question",
                "context",
                "conversation",
                "uuid",
            ]
        ],
        OUT_PATH_REP_DOCS,
    )


if __name__ == "__main__":
    plac.call(main)
