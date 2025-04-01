from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

from dsp_interview_transcripts import logger


def get_min_radius(
    clustered_data: pd.DataFrame,
    k_neighbours: int = 10,
    topic_col: str = "topic",
    embedding_col: str = "norm_embedding",
    metric="cosine",
) -> Tuple[Dict[int, np.ndarray], pd.DataFrame]:
    """
    Calculate the minimum radius that contains `k_neighbours` neighbors for each point in each cluster.
    This is the distance away of the Kth nearest neighbour.

    Args:
        clustered_data (pd.DataFrame): DataFrame containing cluster labels and embeddings. One row per document.
        k_neighbours (int): Number of neighbors to include within the radius. Default is 10.
        topic_col (str): Column name identifying cluster labels. Default is 'topic'.
        embedding_col (str): Column name containing embedding vectors. Default is 'norm_embedding'.

    Returns:
        Tuple[Dict[int, np.ndarray], pd.DataFrame]:
            - Dictionary with cluster labels as keys and arrays of minimum radii for the points in the cluster.
            - Modified DataFrame with an added column indicating the minimum radius for each point to its `k_neighbours`th neighbor.
    """

    clustered_data_copy = clustered_data.copy()

    # Dictionary to store radius distributions for each cluster
    radius_distributions = {}

    # For each cluster, calculate the smallest radius that contains `k_neighbors` neighbors for each point
    for cluster_label in clustered_data_copy[topic_col].unique():
        # Get embeddings for the current cluster
        cluster_points = clustered_data_copy[clustered_data_copy[topic_col] == cluster_label]
        embeddings = np.vstack(cluster_points[embedding_col].values)  # Stack embeddings as an array

        # Calculate pairwise distances within the cluster
        # Because the embeddings are normalized, it shouldn't matter if we use euclidean or cosine? But euclidean is a bit easier to think about/write tests for?
        distances = pairwise_distances(embeddings, metric=metric)

        # Order the matrix so that the 0th column is the distance to itself,
        # 1 column is distance to closest neighbour, 2 column is the distance to the second closest neighbour, etc.
        sorted_distances = np.sort(distances, axis=1)
        # Ensure k_neighbours does not exceed available neighbors
        valid_k = min(k_neighbours, distances.shape[1] - 1)
        radii = sorted_distances[:, valid_k]

        # Store minimum radii for this cluster
        radius_distributions[cluster_label] = radii

        # Add back into the dataframe
        clustered_data_copy.loc[cluster_points.index, f"radius_{k_neighbours}"] = radii

    return radius_distributions, clustered_data_copy


def extract_repr_docs(
    clustered_data: pd.DataFrame, n: int = 10, random_seed: int = 42, user_id_col="conversation"
) -> pd.DataFrame:
    """
    Extract a representative sample of N documents for each topic.

    We do this by finding documents that are in the most dense part or parts of the cluster
    (those docs whose minimum radius to the 10th nearest point is in the first quartile of the distribution
    of such distances for the cluster). We make sure each user is only represented once in the sample,
    then take a random sample of n documents.

    Args:
        clustered_data (pd.DataFrame): DataFrame with cluster data including 'topic', 'conversation', and 'quartile' columns.

    Returns:
        pd.DataFrame: A DataFrame with a sample of documents, one per conversation in each topic,
                      containing up to 10 documents per topic.
    """
    first_quartile_data = clustered_data[clustered_data["quartile"] == "1st"]

    # Get info on the number of conversations (number of users) per topic
    # - so that we know one user isn't dominating
    distinct_conversations = first_quartile_data.groupby("topic")[user_id_col].nunique().reset_index()
    distinct_conversations.columns = ["topic", "distinct_conversations_in_1st_quartile"]
    logger.info(f"Number of distinct conversations per topic: {distinct_conversations}")

    # Keep only one response per user (conversation) in each topic
    unique_conversations = first_quartile_data.drop_duplicates(subset=["topic", user_id_col])
    # Take a random sample of 10
    sampled_data = (
        unique_conversations.groupby("topic")
        .apply(lambda x: x.sample(n=n, random_state=random_seed) if len(x) >= n else x)
        .reset_index(drop=True)
    )

    return sampled_data
