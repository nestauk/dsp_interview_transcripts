# from dsp_interview_transcripts.getters.final import get_summary_table
# from dsp_interview_transcripts.getters.interim import get_data_w_topics
# from dsp_interview_transcripts.getters.interim import get_rep_docs
# from dsp_interview_transcripts.getters.interim import get_topic_names
# from dsp_interview_transcripts.getters.raw import get_raw_transcripts_cleaned
# import os

import pandas as pd


# from dotenv import load_dotenv


# load_dotenv()

# S3_BUCKET = os.environ.get("S3_BUCKET")

rep_docs = pd.read_csv("data/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv")
data = pd.read_csv("data/user_messages_min_len_9_w_sentiment_topics.csv")
data_w_names = pd.read_csv("data/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv")

data = data.rename(columns={"Name": "Topic_name"})

topic_counts = pd.DataFrame(data["Topic"].value_counts()).reset_index()
topic_counts = topic_counts.rename(columns={"count": "N responses in topic"})

data_w_names = data_w_names.rename(
    columns={"Name": "Topic_name", "llama3.2_name": "Name", "llama3.2_description": "Description"}
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

transcripts = pd.read_csv("data/qual_af_transcripts_cleaned.csv")

summary_info = pd.read_csv("data/summary_info.csv")[
    ["Name", "Description", "Top words", "N responses in topic"]
].drop_duplicates()
#    'conversation', 'uuid', 'context','text_clean']]
