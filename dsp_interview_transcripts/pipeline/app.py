
"""
Usage:
```
streamlit run dsp_interview_transcripts/pipeline/app.py
```
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dsp_interview_transcripts import PROJECT_DIR

rep_docs = pd.read_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv")
data = pd.read_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics.csv")
data_w_names = pd.read_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv")

rep_docs = pd.merge(rep_docs, data_w_names[['Cluster', 'llama3.2_name', 'llama3.2_description']], on="Cluster", how="left")

data_viz = pd.merge(data, data_w_names[['Cluster', 'llama3.2_name', 'llama3.2_description']], on="Cluster", how="left")

data_viz['Name'] = data_viz['llama3.2_name'].fillna("None")
data_viz['Description'] = data_viz['llama3.2_description'].fillna("None")

# Set up the Streamlit app layout
st.title("Prevalence and Sentiment Analysis")

# # Sidebar for user input
# st.sidebar.header("Select Options")

# First section: Displaying the side-by-side bar plots
# st.subheader("Side-by-side Bar Plots")

# # Creating the bar plot for the prevalence of each 'Name' using Seaborn
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# # Prevalence of each 'Name'
# sns.countplot(x='Name', data=data_viz, ax=ax[0])
# ax[0].set_title('Prevalence of Each Name')
# ax[0].set_xlabel('Name')
# ax[0].set_ylabel('Count')

# # Percentage of 'sentiment' under each value of 'Name' using Seaborn
# sentiment_percentage = pd.crosstab(data_viz['Name'], data_viz['sentiment'], normalize='index') * 100
# sentiment_percentage = sentiment_percentage.reset_index().melt(id_vars='Name', var_name='sentiment', value_name='percentage')
# sns.barplot(x='Name', y='percentage', hue='sentiment', data=sentiment_percentage, ax=ax[1], estimator=sum)
# ax[1].set_title('Percentage of Sentiment under Each Name')
# ax[1].set_xlabel('Name')
# ax[1].set_ylabel('Percentage')

# # Displaying the plots side by side
# st.pyplot(fig)

# Creating the bar plot for the prevalence of each 'Name' using Seaborn
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Prevalence of each 'Name' with 'Name' on the vertical axis
sns.countplot(y='Name', data=data_viz, ax=ax[0])  # Changed 'x' to 'y' to flip the axis
ax[0].set_title('Prevalence of Each Topic')
ax[0].set_ylabel(None)
ax[0].set_xlabel('Count')

# Percentage of 'sentiment' under each value of 'Name' using Seaborn (stacked)
sentiment_percentage = pd.crosstab(data_viz['Name'], data_viz['sentiment'], normalize='index') * 100
sentiment_percentage = sentiment_percentage.reset_index()

# Plotting the stacked bar chart with 'Name' on the vertical axis
sentiment_percentage.plot(
    kind='barh', stacked=True, x='Name', ax=ax[1], color=sns.color_palette("viridis",3)
)
ax[1].set_title('Percentage of Sentiment under Each Topic')
ax[1].set_ylabel(None)
ax[1].set_xlabel('Percentage')

# Displaying the plots side by side
plt.tight_layout()
st.pyplot(fig)

# Second section: User selects a 'Name' and displays a table
st.subheader("Select a Topic and Display Corresponding Texts")

# Dropdown for selecting a 'Name'
selected_name = st.selectbox("Select a Topic", data_viz['Name'].unique())

if selected_name != "None":
    st.markdown(f"**Description of the selected topic:** \n {data_viz[data_viz['Name'] == selected_name]['Description'].values[0]}")

    st.markdown(f"**Keywords for the selected topic:** \n {rep_docs[rep_docs['llama3.2_name']==selected_name]['Top Words'].drop_duplicates().values[0]}")

# Filtering the dataframe to display the corresponding texts for the selected 'Name'
temp_docs = rep_docs[rep_docs['llama3.2_name'] == selected_name]
filtered_table = temp_docs[['conversation','context', 'text_clean']]
st.table(filtered_table)

# Third section: User selects a 'question' and displays a bar plot
st.subheader("Select a Question and Display Prevalence of Different Topics")

# Dropdown for selecting a 'question'
selected_question = st.selectbox("Select a Question", data_viz['question'].unique())

# Creating a bar plot for the prevalence of 'Name' for the selected question using Seaborn
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.countplot(y='Name', data=data_viz[data_viz['question'] == selected_question], ax=ax2)
ax2.set_title(f'Prevalence of topics for "{selected_question}"')
ax2.set_xlabel(None)
ax2.set_ylabel(None)

# Displaying the bar plot
st.pyplot(fig2)
