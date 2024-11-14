import random

import numpy as np
import pandas as pd
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
from dsp_interview_transcripts import logger
from dsp_interview_transcripts.utils.repr_docs import *


# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DATA_PATH = PROJECT_DIR / "data/user_messages_min_len_9_w_sentiment.csv"

if __name__ == "__main__":

    empty_reduction_model = BaseDimensionalityReduction()

    umap_model = UMAP(
        n_neighbors=15,
        n_components=50,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = TfidfVectorizer(
        stop_words="english",
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
        # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
        "MMR": mmr_model,
        # "POS": pos_model,
    }

    topic_model = BERTopic(
        # Pipeline models
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        umap_model=empty_reduction_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # Hyperparameters
        top_n_words=10,
        verbose=True,
        calculate_probabilities=True,
    )

    df = pd.read_csv(DATA_PATH)

    df["text_clean"] = df["text_clean"].astype(str)
    docs = df["text_clean"].tolist()
    embeddings = SENTENCE_MODEL.encode(docs, show_progress_bar=True)

    embeddings_50d = umap_model.fit_transform(embeddings)

    normalized_embeddings = normalize(embeddings_50d, norm="l2")

    topics, probs = topic_model.fit_transform(docs, normalized_embeddings)

    rep_docs = topic_model.get_representative_docs()

    umap_2d = UMAP(random_state=RANDOM_SEED, n_components=2)
    embeddings_2d = umap_2d.fit_transform(embeddings)

    topic_lookup = topic_model.get_topic_info()[["Topic", "Name"]]

    df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df_vis["topic"] = topics
    df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
    df_vis["doc"] = docs

    df_vis = pd.merge(
        df[["uuid", "conversation", "text_clean", "sentiment", "question", "context"]],
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

    topic_lookup = topic_model.get_topic_info()[["Topic", "Representation"]]

    df_vis = (
        df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
        .drop(columns=["Topic_y"])
        .rename(columns={"Topic_x": "Topic"})
    )

    # save df_vis
    df_vis.to_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics.csv", index=False)

    # save most representative documents
    df_vis_no_noise = df_vis[df_vis["topic"] != -1]
    radius_distributions, clustered_data = get_min_radius(df_vis_no_noise, k_neighbours=10)

    clustered_data["quartile"] = clustered_data.groupby("topic")["radius_10"].transform(
        lambda x: pd.qcut(x, q=4, labels=["1st", "2nd", "3rd", "greater than 3rd"])
    )
    repr_docs = extract_repr_docs(clustered_data)

    repr_docs[
        ["Topic", "Representation", "text_clean", "sentiment", "question", "context", "conversation", "uuid"]
    ].to_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv", index=False)
