from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import altair as alt
import pandas as pd
import plac

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import S3_BUCKET
from dsp_interview_transcripts import config
from dsp_interview_transcripts import logger
from dsp_interview_transcripts.getters.data_getters import save_to_s3
from dsp_interview_transcripts.getters.data_getters import upload_file_to_s3
from dsp_interview_transcripts.getters.interim import get_data_w_topics
from dsp_interview_transcripts.getters.interim import get_rep_docs
from dsp_interview_transcripts.getters.interim import get_topic_names


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


@plac.annotations(production=("Run in production mode if True, otherwise in test mode", "flag", "production"))
def main(production: bool = False):

    PROJECT = config["project"]

    if production:
        OUTPUT_PATH_FULL_DATA = f"{PROJECT}/" + config["prod_paths"]["final_full_data_s3_path"]
        OUTPUT_PATH_SUMMARY = f"{PROJECT}/" + config["prod_paths"]["final_summary_info_s3_path"]
        LOCAL_OUTPUTS = PROJECT_DIR / "outputs/final"
        S3_FIGURES = f"{PROJECT}/" + config["prod_paths"]["final_figures_s3_path"]
    else:
        OUTPUT_PATH_FULL_DATA = f"{PROJECT}/" + config["test_paths"]["final_full_data_s3_path"]
        OUTPUT_PATH_SUMMARY = f"{PROJECT}/" + config["test_paths"]["final_summary_info_s3_path"]
        LOCAL_OUTPUTS = PROJECT_DIR / "outputs/final/test"
        S3_FIGURES = f"{PROJECT}/" + config["test_paths"]["final_figures_s3_path"]

    # Create the local output directory if it doesn't exist
    LOCAL_OUTPUTS.mkdir(parents=True, exist_ok=True)

    rep_docs = get_rep_docs(production=production)
    data = get_data_w_topics(production=production)
    data_w_names = get_topic_names(production=production)

    topic_counts = pd.DataFrame(data["Topic"].value_counts()).reset_index()
    topic_counts = topic_counts.rename(columns={"count": "N responses in topic"})

    data_w_names = pd.merge(data_w_names, topic_counts, left_on="Topic", right_on="Topic", how="left")

    rep_docs = pd.merge(
        rep_docs,
        data_w_names[["Topic", "llama3.2_name", "llama3.2_description", "N responses in topic"]],
        on="Topic",
        how="left",
    )
    rep_docs = rep_docs.rename(
        columns={
            "Name": "keyword_name",
            "Representation": "Top words",
            "llama3.2_name": "Name",
            "llama3.2_description": "Description",
        }
    )

    save_to_s3(
        S3_BUCKET,
        rep_docs[
            [
                "Name",
                "Description",
                "keyword_name",
                "Top words",
                "N responses in topic",
                "conversation",
                "uuid",
                "text_clean",
                "context",
                "sentiment",
            ]
        ],
        OUTPUT_PATH_SUMMARY,
    )

    data = data.rename(columns={"Name": "keyword_name", "Representation": "Top words"})
    data_viz = (
        data.merge(rep_docs[["Topic", "Name", "Description"]], on="Topic", how="left")
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
            "keyword_name",
            "Top words",
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
            "Top words": "Topic Top Words",
        }
    ).sort_values(["conversation", "timestamp"])

    logger.info("Saving output table...")

    save_to_s3(S3_BUCKET, final_df, OUTPUT_PATH_FULL_DATA)

    # Visualise clusters
    logger.info("Saving figures...")

    fig = create_scatterplot(
        data_viz,
    )
    fig.save(PROJECT_DIR / "outputs/scatter_coloured_by_topic.html")
    upload_file_to_s3(
        S3_BUCKET,
        f"{PROJECT_DIR}/outputs/scatter_coloured_by_topic.html",
        f"{S3_FIGURES}/scatter_coloured_by_topic.html",
    )

    fig_questions = create_scatterplot(
        data_viz=data_viz,
        color="question:N",
    )
    fig_questions.save(PROJECT_DIR / "outputs/scatter_coloured_by_question.html")
    upload_file_to_s3(
        S3_BUCKET,
        f"{PROJECT_DIR}/outputs/scatter_coloured_by_question.html",
        f"{S3_FIGURES}/scatter_coloured_by_question.html",
    )

    fig_sentiment = create_scatterplot(
        data_viz=data_viz,
        color="sentiment:N",
        domain=["Negative", "Neutral", "Positive"],
        range_=["red", "gray", "green"],
    )
    fig_sentiment.save(PROJECT_DIR / "outputs/scatter_coloured_by_sentiment.html")
    upload_file_to_s3(
        S3_BUCKET,
        f"{PROJECT_DIR}/outputs/scatter_coloured_by_sentiment.html",
        f"{S3_FIGURES}/scatter_coloured_by_sentiment.html",
    )


if __name__ == "__main__":
    plac.call(main)
