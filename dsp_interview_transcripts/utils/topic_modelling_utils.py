import pandas as pd
from typing import List

def concat_topics_probabilities(
    input_df: pd.DataFrame, topics: List[str], docs, probs, axis=1
):
    """Bring together the original dataframe of docs, and the probabilities and topic
    labels from the topic model.
    """
    df = pd.DataFrame(
        {
            "doc_index": range(len(docs)),
            "topic": topics,
            "probability": probs.max(axis=axis),
        }
    )

    output_df = pd.concat(
        [input_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1
    )

    return output_df