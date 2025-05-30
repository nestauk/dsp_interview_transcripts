from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

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
    logger.info("Embedding user messages...")

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
        max_df=0.7,
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


def get_proportion_noise(topics: List[int], save: bool = False, outpath: Optional[str] = None) -> float:
    """Calculates the proportion of noise in a list of topics.

    Args:
        topics (List[int]): List of topic labels of datapoints (i.e. not the unique topics; the labels for all data)
        save (bool, optional): Do you want to save the proportion of noise to a file? Defaults to False.
        outpath (Optional[str], optional): Local path for saving the proportion of noise. Defaults to None.

    Returns:
        float: Proportion of noise in the list of topics
    """
    total_elements = len(topics)
    count_noise = topics.count(-1)
    proportion_noise = count_noise / total_elements
    logger.info(f"{proportion_noise}")
    if save:
        with open(outpath, "w") as f:
            f.write(str(proportion_noise))
    return proportion_noise
