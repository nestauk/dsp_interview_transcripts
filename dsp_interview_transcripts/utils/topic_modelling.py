from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from dsp_interview_transcripts import logger


def embed_docs(
    docs: List[str], model: Optional[SentenceTransformer] = None, save: bool = False, outpath: str = "embeddings.npy"
) -> Tuple[List[str], np.ndarray]:
    """
    Embed a list of documents using a SentenceTransformer model.
    Wrote this function just to save a tiny bit of repeated code.

    Args:
        docs (List[str]): List of documents to embed.
        model (Optional[SentenceTransformer]): Pre-trained SentenceTransformer model. If None, defaults to "all-MiniLM-L6-v2".
        save (bool): Whether to save the embeddings to a file.
        outpath (str, optional): Path to save the embeddings. Defaults to "embeddings.npy".

    Returns:
        Tuple[List[str], np.ndarray]: Original documents and their embeddings.
    """

    logger.info("Embedding user messages...")

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(docs, show_progress_bar=True)
    if save:
        np.save(outpath, embeddings)
    return docs, embeddings


def init_topic_model(
    stop_words: List[str],
    min_cluster_size: int,
    hdbscan_selection_method: str,
    embedding_model: SentenceTransformer,
    seed: int = 42,
    empty_reduction: bool = False,
) -> Tuple[BERTopic, TfidfVectorizer, Dict[str, object]]:
    """
    Initialize a BERTopic model with specified configurations.

    The vectorizer and representation models are returned alongside the topic model
    because this way they can be used for reassigning the noise cluster.

    This function exists just so that it's easy to vary the key hyperparameters
    we're interested in, without having to repeat the code that sets all the other hyperparams.

    Args:
        stop_words (List[str]): List of stopwords to use in the vectorizer. Can be e.g. "english" or a custom list.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        hdbscan_selection_method (str): Cluster selection method for HDBSCAN - "eom" or "leaf".
        embedding_model (SentenceTransformer): SentenceTransformer model for embeddings.
        seed (int, optional): Random seed for UMAP. Defaults to 42.
        empty_reduction (bool, optional): Whether to use an empty dimensionality reduction model. Defaults to False. This might be useful if you want to normalise the embeddings or any of your other own transformations.

    Returns:
        Tuple[BERTopic, TfidfVectorizer, Dict[str, object]]: Initialized BERTopic model, vectorizer model, and representation model.
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


def get_proportion_noise(topics: List[int], save: bool, outpath: Optional[str] = None) -> float:
    """
    Calculate the proportion of noise (-1) in the topic assignments.
    It's useful to be able to do this as an indicator of how the model is performing. Bad models
    might have either a really high proportion of noise, or a really low proportion.

    Args:
        topics (List[int]): List of topic assignments, where -1 represents noise.
        save (bool): Whether to save the proportion of noise to a file.
        outpath (Optional[str], optional): Path to save the noise proportion. Defaults to None.

    Returns:
        float: Proportion of noise in the topic assignments.
    """
    total_elements = len(topics)
    count_noise = topics.count(-1)
    proportion_noise = count_noise / total_elements
    logger.info(f"Proportion noise: {proportion_noise}")
    if save:
        with open(outpath, "w") as f:
            f.write(str(proportion_noise))
    return proportion_noise
