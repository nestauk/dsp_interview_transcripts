from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import altair as alt
import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


OUTPUT_DIR = PROJECT_DIR / "outputs/final"
OUTPUT_PATH_FULL_DATA = OUTPUT_DIR / "final_df.csv"
OUTPUT_PATH_SUMMARY = OUTPUT_DIR / "summary_info.csv"

# Create the output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

opacity_condition = alt.condition(alt.datum.Name == "None", alt.value(0.1), alt.value(0.6))


def create_scatterplot(
    data_viz: pd.DataFrame,
    color: str = "Name:N",
    tooltip: List[str] = ["Name:N", "Description:N", "question:N", "text_clean:N"],
    domain: Optional[List[Union[str, int]]] = None,
    range_: Optional[List[str]] = None,
) -> alt.Chart:
    """
    Display texts as a scatterplot.

    domain and range_ are used for specifying a 3-way colour scale when the
    texts should be coloured by sentiment.

    Args:
        data_viz (pd.DataFrame): The DataFrame containing data to visualize. Must include columns for x and y coordinates,
            along with fields specified in `color` and `tooltip`.
        color (str, optional): Encoding specification for the color channel. Defaults to "Name:N".
        tooltip (List[str], optional): List of fields to display as tooltips. Defaults to ["Name:N", "Description:N", "question:N", "text_clean:N"].
        domain (Optional[List[Union[str, int]]], optional): Custom domain values for the color scale, defining specific categories.
            Defaults to None.
        range_ (Optional[List[str]], optional): Custom color range for the color scale, corresponding to the domain values.
            Defaults to None.

    Returns:
        alt.Chart: An Altair chart object representing the scatterplot, with specified color, tooltips, and interactivity.
    """

    if domain is not None and range_ is not None:
        color = alt.Color(color, scale=alt.Scale(domain=domain, range=range_))
    else:
        color = alt.Color(color)

    fig = (
        alt.Chart(data_viz)
        .mark_circle(size=50)
        .encode(
            x=alt.X(
                "x:Q",
                axis=alt.Axis(ticks=False, labels=False, title=None, grid=False),
            ),
            y=alt.Y(
                "y:Q",
                axis=alt.Axis(ticks=False, labels=False, title=None, grid=False),
            ),
            color=color,
            opacity=opacity_condition,  # Ensure opacity_condition is defined elsewhere
            tooltip=tooltip,
        )
        .properties(width=900, height=600)
        .interactive()
    )

    return fig


if __name__ == "__main__":
    rep_docs = pd.read_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv")
    data = pd.read_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics.csv")
    data_w_names = pd.read_csv(
        PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv"
    )

    topic_counts = pd.DataFrame(data["Cluster"].value_counts()).reset_index()
    topic_counts = topic_counts.rename(columns={"count": "N responses in topic"})

    data_w_names = data_w_names.rename(columns={"llama3.2_name": "Name", "llama3.2_description": "Description"})
    data_w_names = pd.merge(data_w_names, topic_counts, left_on="Cluster", right_on="Cluster", how="left")

    data_w_names[["Name", "Description", "Top Words", "N responses in topic"]].to_csv(OUTPUT_PATH_SUMMARY, index=False)

    rep_docs = pd.merge(
        rep_docs, data_w_names[["Cluster", "Name", "Description", "N responses in topic"]], on="Cluster", how="left"
    )

    data_viz = (
        data.merge(data_w_names[["Cluster", "Name", "Description"]], on="Cluster", how="left")
        .assign(Name=lambda df: df["Name"].fillna("None"))
        .assign(Description=lambda df: df["Description"].fillna("None"))
    )

    # Create binary column to indicate whether the user response is representative of the topic
    merged_df = (
        data_viz.merge(
            rep_docs[["conversation", "uuid", "Name"]], on=["conversation", "uuid", "Name"], how="left", indicator=True
        )
        .assign(Representative_of_topic=lambda df: (df["_merge"] == "both").astype(int))
        .drop(columns=["_merge"])
    )

    final_df = merged_df[
        [
            "Name",
            "Description",
            "Top Words",
            "Representative_of_topic",
            "question",
            "context",
            "text_clean",
            "sentiment",
            "conversation",
            "uuid",
            "timestamp",
        ]
    ]
    final_df = final_df.rename(
        columns={
            "text_clean": "user_response",
            "question": "probable question",
            "sentiment": "predicted_sentiment",
            "Name": "Topic Name",
            "Description": "Topic Description",
            "Top Words": "Topic Top Words",
        }
    )

    logger.info("Saving output table...")
    final_df.sort_values(["conversation", "timestamp"]).to_csv(OUTPUT_PATH_FULL_DATA, index=False)

    # Visualise clusters
    logger.info("Saving figures...")

    fig = create_scatterplot(
        data_viz,
    )
    fig.save(PROJECT_DIR / "outputs/scatter_coloured_by_topic.html")

    fig_questions = create_scatterplot(
        data_viz=data_viz,
        color="question:N",
    )
    fig_questions.save(PROJECT_DIR / "outputs/scatter_coloured_by_question.html")

    fig_sentiment = create_scatterplot(
        data_viz=data_viz,
        color="sentiment:N",
        domain=["Negative", "Neutral", "Positive"],
        range_=["red", "gray", "green"],
    )
    fig_sentiment.save(PROJECT_DIR / "outputs/scatter_coloured_by_sentiment.html")
