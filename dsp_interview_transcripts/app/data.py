import pandas as pd

from dsp_interview_transcripts.getters.interim import get_data_w_topics
from dsp_interview_transcripts.getters.interim import get_rep_docs
from dsp_interview_transcripts.getters.interim import get_topic_names
from dsp_interview_transcripts.getters.raw import get_raw_transcripts_cleaned


rep_docs = get_rep_docs(production=True)
data = get_data_w_topics(production=True)
data_w_names = get_topic_names(production=True)

topic_counts = pd.DataFrame(data["Cluster"].value_counts()).reset_index()
topic_counts = topic_counts.rename(columns={"count": "N responses in topic"})

data_w_names = data_w_names.rename(columns={"llama3.2_name": "Name", "llama3.2_description": "Description"})
data_w_names = pd.merge(data_w_names, topic_counts, left_on="Cluster", right_on="Cluster", how="left")

data_viz = (
    data.merge(data_w_names[["Cluster", "Name", "Description"]], on="Cluster", how="left")
    .assign(Name=lambda df: df["Name"].fillna("None"))
    .assign(Description=lambda df: df["Description"].fillna("None"))
)

transcripts = get_raw_transcripts_cleaned(production=True)
