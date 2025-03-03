"""
Example usage:
```
python dsp_interview_transcripts/pipeline/topic_modelling.py -s leaf -c 50 -r embeddings
```
or
```
python dsp_interview_transcripts/pipeline/topic_modelling.py -s eom -c 50 -r probabilities --production
```
"""
import os
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

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

MIN_CLUSTER_SIZE = 20
MIN_LEN = config["min_length"]
PROJECT = config["project"]
# directory for local outputs
OUTPATH = f"{PROJECT_DIR}/outputs/{PROJECT}/"
os.makedirs(OUTPATH, exist_ok=True)


@plac.opt("selection", "Cluster selection method ('eom' or 'leaf')", type=str, abbrev="s")
@plac.opt("min_cluster_size", "Minimum cluster size for HDBSCAN", type=int, abbrev="c")
@plac.opt(
    "reduction_strategy",
    "Strategy for reducing outliers ('embeddings', 'probabilities', 'distributions', 'ctfidf')",
    type=str,
    abbrev="r",
)
@plac.annotations(production=("Run script in production mode if True, otherwise in test mode", "flag", "p"))
def main(
    selection="leaf", min_cluster_size=MIN_CLUSTER_SIZE, reduction_strategy="embeddings", production: bool = False
):

    if reduction_strategy == "ctfidf":
        reduction_strategy = "c-tf-idf"

    if production:
        OUT_PATH_FULL_DATA = f"{PROJECT}/" + config["prod_paths"]["interim_data_w_topics_s3_path"].format(
            MIN_LEN=MIN_LEN
        )
        OUT_PATH_REP_DOCS = f"{PROJECT}/" + config["prod_paths"]["interim_representative_docs_s3_path"].format(
            MIN_LEN=MIN_LEN
        )
    else:
        OUT_PATH_FULL_DATA = f"{PROJECT}/" + config["test_paths"]["interim_data_w_topics_s3_path"].format(
            MIN_LEN=MIN_LEN
        )
        OUT_PATH_REP_DOCS = f"{PROJECT}/" + config["test_paths"]["interim_representative_docs_s3_path"].format(
            MIN_LEN=MIN_LEN
        )

    user_messages = get_cleaned_data(production=production)

    docs = user_messages["text_clean"].tolist()
    docs, embeddings = embed_docs(docs, SENTENCE_MODEL, save=False)

    topic_model, vectorizer_model, representation_model = init_topic_model(
        stop_words="english",
        min_cluster_size=min_cluster_size,
        hdbscan_selection_method=selection,
        embedding_model=MODEL_NAME,
        seed=RANDOM_SEED,
        empty_reduction=False,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    print(f"Unique topics before reduction: {set(topics)}")
    print(f"Reduction strategy: {reduction_strategy}")

    if reduction_strategy == "probabilities":
        new_topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy=reduction_strategy)
    else:
        new_topics = topic_model.reduce_outliers(docs, topics, strategy=reduction_strategy)

    topic_model.update_topics(
        docs,
        topics=new_topics,
        top_n_words=10,
        n_gram_range=(1, 3),
        vectorizer_model=vectorizer_model,
        ctfidf_model=None,
        representation_model=representation_model,
    )

    # Default BERTopic scatterplot
    fig = topic_model.visualize_documents(docs, embeddings=embeddings)
    output_path = f"{OUTPATH}bertopic_visualization_selection_{selection}_min_length_{MIN_LEN}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.html"
    fig.write_html(output_path)

    # Default BERTopic summary info
    summary_info = topic_model.get_topic_info()
    # save locally
    topic_info_path = f"{OUTPATH}bertopic_topic_info_selection_{selection}_min_length_{MIN_LEN}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.csv"
    summary_info.to_csv(topic_info_path, index=False)
    # save to s3
    if production:
        bertopic_summary_outpath = f"{PROJECT}/" + "interim/bertopic_topic_info.csv"
    else:
        bertopic_summary_outpath = f"{PROJECT}/" + "test/interim/bertopic_topic_info.csv"
    save_to_s3(
        S3_BUCKET,
        summary_info,
        bertopic_summary_outpath,
    )

    # What proportion is noise?
    proportion_noise = get_proportion_noise(
        new_topics,
        save=True,
        outpath=f"{OUTPATH}noise_prop_selection_{selection}_min_length_{MIN_LEN}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.txt",
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

    df_vis["embedding"] = list(embeddings)

    logger.info("Saving data...")
    save_to_s3(S3_BUCKET, df_vis, OUT_PATH_FULL_DATA)

    # save most representative documents
    df_vis_no_noise = df_vis[df_vis["topic"] != -1]
    _, clustered_data = get_min_radius(df_vis_no_noise, k_neighbours=10, embedding_col="embedding")

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
