import numpy as np

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

from dsp_interview_transcripts import logger


def embed_docs(docs, model, save: bool, outpath="embeddings.npy"):
    logger.info("Embedding user messages...")

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(docs, show_progress_bar=True)
    if save:
        np.save(outpath, embeddings)
    return docs, embeddings


def init_topic_model(
    stop_words, min_cluster_size, hdbscan_selection_method, embedding_model, seed=42, empty_reduction=False
):

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


def get_proportion_noise(topics, save: bool, outpath=None):
    total_elements = len(topics)
    count_noise = topics.count(-1)
    proportion_noise = count_noise / total_elements
    logger.info(f"{proportion_noise}")
    # noise_path = f"{OUTPATH}noise_prop_selection_{selection}_min_length_{min_length}_min_cluster_{min_cluster_size}_red_{reduction_strategy}.txt"
    if save:
        with open(outpath, "w") as f:
            f.write(str(proportion_noise))
    return proportion_noise
