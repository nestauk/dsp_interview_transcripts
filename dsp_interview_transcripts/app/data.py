import pandas as pd


DATA_DIR = "data/"

FILENAME_DATA = "user_messages_min_len_9_w_sentiment_topics.csv"
FILENAME_DATA_CLUSTER_NAMES = "user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv"
FILENAME_SUMMARY_INFO = "summary_info.csv"
FILENAME_TRANSCRIPTS = "qual_af_transcripts_cleaned.csv"

data = pd.read_csv(f"{DATA_DIR}{FILENAME_DATA}").rename(columns={"Name": "Topic_name"})
data_w_names = pd.read_csv(f"{DATA_DIR}{FILENAME_DATA_CLUSTER_NAMES}").rename(
    columns={"Name": "Topic_name", "llama3.2_name": "Name", "llama3.2_description": "Description"}
)

transcripts = pd.read_csv(f"{DATA_DIR}{FILENAME_TRANSCRIPTS}")

summary_info = pd.read_csv(f"{DATA_DIR}{FILENAME_SUMMARY_INFO}")[
    ["Name", "Description", "Top words", "N responses in topic"]
].drop_duplicates()

topic_counts = (
    pd.DataFrame(data["Topic"].value_counts()).reset_index().rename(columns={"count": "N responses in topic"})
)

data_w_names = pd.merge(data_w_names, topic_counts, left_on="Topic", right_on="Topic", how="left")

data_viz = (
    data.merge(data_w_names[["Topic", "Name", "Description"]], on="Topic", how="left")
    .assign(Name=lambda df: df["Name"].fillna("None"))
    .assign(Description=lambda df: df["Description"].fillna("None"))
)

names = data_viz["Name"].unique().tolist()

# 0.2 for the noise cluster, otherwise 0.8
data_viz["opacity"] = data_viz["Name"].apply(lambda Name: 0.8 if Name in names[1:] else 0.2)
