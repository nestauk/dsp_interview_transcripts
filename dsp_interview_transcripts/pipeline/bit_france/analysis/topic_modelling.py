"""
Example usage:
```
python dsp_interview_transcripts/pipeline/bit_france/analysis/topic_modelling.py -s eom -l 9 -c 50
```
"""

import os
import random

import nltk
import numpy as np
import pandas as pd
import plac
import torch

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
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


nltk.download("stopwords")
nltk.download("punkt")

french_stopwords = stopwords.words("french")

# Set random seeds
RANDOM_SEED = 42
from numpy import random as npr


npr.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# SENTENCE_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model_name = "dangvantuan/sentence-camembert-large"
SENTENCE_MODEL = SentenceTransformer(model_name)


def prep_data(data, min_length):
    # We have to deduplicate because there seems to be at least one duplicate interview
    data = data.drop_duplicates(subset=["text"])
    logger.info(f"N records after deduplication: {len(data)}")

    logger.info(data["role"].value_counts())

    # Isolate just the interviewees
    speaker_data = data[data["role"] == "informant"]

    speaker_data["word_count"] = speaker_data["text"].apply(lambda x: len(x.split()))

    speaker_data_filtered = speaker_data[speaker_data["word_count"] > min_length]

    return speaker_data_filtered


def embed_docs(speaker_data_filtered, min_length, outpath, model):
    docs = speaker_data_filtered["text"].tolist()
    logger.info("Embedding user messages...")
    embeddings = model.encode(docs, show_progress_bar=True)
    embeddings_path = f"{outpath}embeddings_min_length_{min_length}.npy"
    np.save(embeddings_path, embeddings)
    return docs, embeddings


@plac.opt("selection", "Cluster selection method ('eom' or 'leaf')", type=str)
@plac.opt("min_length", "Minimum length of word count for filtering", type=int, abbrev="l")
@plac.opt("min_cluster_size", "Minimum cluster size for HDBSCAN", type=int, abbrev="c")
@plac.opt(
    "reduction_strategy",
    "Strategy for reducing outliers ('embeddings', 'probabilities', 'distributions', 'ctfidf')",
    type=str,
    abbrev="r",
)
def main(selection="eom", min_length=5, min_cluster_size=50, reduction_strategy="embeddings"):

    if reduction_strategy == "ctfidf":
        reduction_strategy = "c-tf-idf"

    maquettes_df = pd.read_csv(f"{PROJECT_DIR}/data/bit_france/converted/maquettes_df.csv")

    professions = ["Décideurs", "Salariés", "Elus"]

    for profession in professions:
        logger.info(f"Processing the interviews of the {profession} group...")
        OUTPATH = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs/{profession}/"
        os.makedirs(OUTPATH, exist_ok=True)

        data = maquettes_df[maquettes_df["profession"] == profession]

        logger.info(f"N records for {profession}: {len(data)}")

        speaker_data_filtered = prep_data(data, min_length)

        docs, embeddings = embed_docs(
            speaker_data_filtered, min_length=min_length, outpath=OUTPATH, model=SENTENCE_MODEL
        )

        umap_model = UMAP(
            n_neighbors=15,
            n_components=50,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_SEED,
        )

        hdbscan_model = HDBSCAN(
            min_samples=5,
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method=selection,
            prediction_data=True,
        )

        vectorizer_model = TfidfVectorizer(
            stop_words=french_stopwords,
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

        topic_model = BERTopic(
            # Pipeline models
            embedding_model=model_name,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            # Hyperparameters
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True,
        )

        topics, probs = topic_model.fit_transform(docs, embeddings)

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
        output_path = f"{OUTPATH}bertopic_visualization_selection_{selection}_min_length_{min_length}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.html"
        fig.write_html(output_path)

        # Default BERTopic summary info
        topic_info = topic_model.get_topic_info()
        topic_info_path = f"{OUTPATH}bertopic_topic_info_selection_{selection}_min_length_{min_length}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.csv"
        topic_info.to_csv(topic_info_path, index=False)

        # What proportion is noise?
        total_elements = len(new_topics)
        count_noise = new_topics.count(-1)
        proportion_noise = count_noise / total_elements
        logger.info(f"{proportion_noise}")
        noise_path = f"{OUTPATH}noise_prop_selection_{selection}_min_length_{min_length}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.txt"
        with open(noise_path, "w") as f:
            f.write(str(proportion_noise))

        speaker_data_filtered["embeddings"] = embeddings.tolist()
        speaker_data_filtered["topic"] = new_topics

        data_out_path = f"{OUTPATH}speaker_data_topics_selection_{selection}_min_length_{min_length}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.csv"
        speaker_data_filtered.to_csv(data_out_path, index=False)

        # Check for topics that have a low median word count (although if min_len is high, there won't be any)
        check_word_count = speaker_data_filtered.groupby("topic").agg(
            median_word_count=("word_count", "median"), unique_file_names=("file_name", "nunique")
        )
        logger.info(check_word_count)
        word_count_path = f"{OUTPATH}topic_median_word_count_selection_{selection}_min_length_{min_length}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.csv"
        check_word_count.to_csv(word_count_path, index=False)


if __name__ == "__main__":
    plac.call(main)
