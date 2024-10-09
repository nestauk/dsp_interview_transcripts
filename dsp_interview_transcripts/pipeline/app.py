
"""
Usage:
```
streamlit run dsp_interview_transcripts/pipeline/app.py
```
"""

import streamlit as st
import pandas as pd
import altair as alt
import re

from dsp_interview_transcripts import PROJECT_DIR

QUESTIONS = ["What, if anything, do you know about the Boiler Upgrade Scheme? If you don't know anything about the scheme, just give it your best guess.",
             "What, if anything, do you know about the process of applying for Boiler Upgrade Scheme funding?",
             "What do you think are the eligibility requirements for someone to use this scheme?",
             "How would you go about finding out more about the Boiler Upgrade Scheme",
             "What do you think about there being eligiblity requirements for a scheme like this?",
             "As a homeowner, where do you see yourself in relation to the eligibility requirements?",
             "What, if any, type of work do you think needs to be done to a house to replace fossil fuel heating systems?",
             "What types of home upgrades would you consider getting done to your house to improve the efficiency of your heating system?",
             "What types of work to your house wouldn't you consider?",
             "What are some energy-efficient heating systems that you could consider, apart from the one currently in use at your home?",
             "Is there anything we've talked about you'd like to discuss further?"]

# Define the app layout
st.title("Topic Modeling Visualization")

INTERVIEW_SECTIONS = [f"interview_q_{i}" for i in range(-1, 10) if i!=4]
DATA_SOURCES = ["user_messages",
                "q_and_a",
                ] + INTERVIEW_SECTIONS

# Input options
data_source = st.selectbox("Select Data Source", DATA_SOURCES)

if data_source in INTERVIEW_SECTIONS:
    index = data_source.split("_")[-1]
    if index=="-1":
        st.info("These responses could not be matched to a question from the prompt.")
    else:
        st.info(QUESTIONS[int(index)])

minimum_cluster_size = st.selectbox("Select Minimum Cluster Size", [10, 50, 100])

# Validation: Prevent selection of '100' and 'home_upgrades_user' together
if data_source in INTERVIEW_SECTIONS and minimum_cluster_size > 10:
    st.warning("For one of the 'interview_q' sources, you can only select a min. cluster size of 10. Please select a different combination.")
else:

    # Filepath based on user inputs
    file_path = PROJECT_DIR / f"outputs/{data_source}_cluster_size_{minimum_cluster_size}_vis.csv"

    # Load the CSV data
    @st.cache
    def load_data(file_path):
        return pd.read_csv(file_path)

    # Load the data and display it
    if file_path.exists():
        df_vis = load_data(file_path)
    else:
        st.error(f"File not found: {file_path}")

    # Altair plotting
    if not df_vis.empty:
        # Define opacity condition
        opacity_condition = alt.condition(
            alt.datum.topic == -1, alt.value(0.1), alt.value(0.4)
        )

        # Create plot
        fig = (
            alt.Chart(df_vis)
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
                color=alt.Color("Name:N"),
                opacity=opacity_condition,
                tooltip=["Name:N", "doc:N"],
            )
            .properties(width=900, height=600)
            .interactive()
        )

        # Display the plot in the Streamlit app
        st.altair_chart(fig, use_container_width=True)
