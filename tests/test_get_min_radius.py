import numpy as np
import pandas as pd
import pytest

from dsp_interview_transcripts.utils.repr_docs import get_min_radius


@pytest.fixture
def dummy_data():
    """Dummy dataset with 2 clusters and 2D embeddings."""
    return pd.DataFrame(
        {
            "topic": [0, 0, 0, 0, 1, 1, 1, 1],
            "norm_embedding": [
                np.array([0, 0]),  # Points in cluster 0
                np.array([1, 0]),
                np.array([0, 1]),
                np.array([1, 1]),
                np.array([4, 4]),  # Points in cluster 1
                np.array([4, 9]),
                np.array([9, 9]),
                np.array([16, 16]),
            ],
        }
    )


def test_get_min_radius(dummy_data):

    k_neighbours = 2

    # Run the function
    radius_distributions, result_df = get_min_radius(
        dummy_data, k_neighbours=k_neighbours, topic_col="topic", embedding_col="norm_embedding"
    )

    # Test radius_distributions dictionary structure and values
    for cluster_label, radii in radius_distributions.items():
        assert len(radii) == 4
        if cluster_label == 0:
            # Cluster 0 points are close to each other; expect radius to be around 1
            expected_radii = np.array([1, 1, 1, 1])
        else:
            # Cluster 1 points are further away from each other
            expected_radii = np.array([7.07106781, 5, 7.07106781, 13.89244399])
        np.testing.assert_almost_equal(radii, expected_radii, decimal=1)

    # Test the result DataFrame has the expected radius column
    assert f"radius_{k_neighbours}" in result_df.columns
    assert len(result_df) == len(dummy_data)  # Should have the same length as input data

    # Check values in the radius column match the expected radii
    for cluster_label, group in result_df.groupby("topic"):
        radii = group[f"radius_{k_neighbours}"].values
        if cluster_label == 0:
            expected_radii = np.array([1, 1, 1, 1])
        else:
            expected_radii = np.array([7.07106781, 5, 7.07106781, 13.89244399])
        np.testing.assert_almost_equal(radii, expected_radii, decimal=1)
