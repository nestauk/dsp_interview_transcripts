
"""
Usage:
```
streamlit run dsp_interview_transcripts/pipeline/app.py
```
"""
import ast
import streamlit as st
import pandas as pd
import altair as alt

from dsp_interview_transcripts import PROJECT_DIR

# Define the app layout
st.title("Topic Modeling Visualization")

INTERVIEW_SECTIONS = [f"interview_q_{i}" for i in range(-1, 10) if i!=4]
DATA_SOURCES = ["user_messages",
                "q_and_a",
                "interviews_chunked",
                ] + INTERVIEW_SECTIONS

# Input options
data_source = st.selectbox("Select Data Source", DATA_SOURCES)
minimum_cluster_size = st.selectbox("Select Minimum Cluster Size", [5, 10, 20, 50, 100])

# Validation: Prevent selection of '100' and 'home_upgrades_user' together
if data_source in INTERVIEW_SECTIONS and minimum_cluster_size > 10:
    st.warning("For one of the 'interview_q' sources, you can only select a min. cluster size of 10. Please select a different combination.")
elif data_source == "interviews_chunked" and minimum_cluster_size > 20:
    st.warning("For 'interviews_chunked', you can only select a min. cluster size of 20. Please select a different combination.")
else:

    # Filepath based on user inputs
    vis_file_path = PROJECT_DIR / f"outputs/{data_source}_cluster_size_{minimum_cluster_size}_vis.csv"
    topic_file_path = PROJECT_DIR / f"outputs/{data_source}_cluster_size_{minimum_cluster_size}_topic_info_with_names_descriptions.csv"
    
    # Load the CSV data
    @st.cache
    def load_data(file_path):
        return pd.read_csv(file_path)

    # Load the data and display it
    if vis_file_path.exists():
        df_vis = load_data(vis_file_path)
    else:
        st.error(f"File not found: {vis_file_path}")
        
    if topic_file_path.exists():
        topic_info = load_data(topic_file_path)
        topic_info['Representative_Docs'] = topic_info['Representative_Docs'].apply(ast.literal_eval)
    else:
        st.error(f"File not found: {topic_file_path}")
    
    # Altair plotting
    if not df_vis.empty:
        
        topic_list = list(df_vis["topic"].unique())
        
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
                tooltip=["Name:N", "Topic:N","doc:N"],
            )
            .properties(width=900, height=600)
            .interactive()
        )

        # Display the plot in the Streamlit app
        st.altair_chart(fig, use_container_width=True)
        
        selected_topic = st.selectbox("Select topic", topic_list)
        
        st.table(topic_info[topic_info["Topic"] == selected_topic][["llama3.2_name", "llama3.2_description", "Name", "MMR"]])
        
        st.table(topic_info[topic_info["Topic"] == selected_topic][["Representative_Docs"]].explode("Representative_Docs"))
