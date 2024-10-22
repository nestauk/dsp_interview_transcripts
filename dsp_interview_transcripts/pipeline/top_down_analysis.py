import altair as alt
from bertopic import BERTopic
import numpy as np
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from openpyxl import Workbook

from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    # OpenAI,
    PartOfSpeech,
)

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

from dsp_interview_transcripts import PROJECT_DIR, logger

pd.set_option('display.max_colwidth', 500)

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# PyTorch seed (used by SentenceTransformer)
import torch

torch.manual_seed(RANDOM_SEED)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
            n_neighbors=15,
            n_components=50,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_SEED,
            )

hdbscan_model = HDBSCAN(
                min_cluster_size=5, # setting this really small because some questions don't have many responses
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

vectorizer_model = CountVectorizer(
                stop_words="english",
                min_df=1,
                max_df=0.85,
                ngram_range=(1, 3),
            )

# KeyBERT
keybert_model = KeyBERTInspired()

# MMR
mmr_model = MaximalMarginalRelevance(diversity=0.3)


# All representation models
representation_model = {
                "KeyBERT": keybert_model,
                # "OpenAI": openai_model,  # Uncomment if you will use OpenAI
                "MMR": mmr_model,
                # "POS": pos_model,
            }

topic_model = BERTopic(
                # Pipeline models
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                representation_model=representation_model,
                # Hyperparameters
                top_n_words=10,
                verbose=True,
                calculate_probabilities=True,
            )

class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")
    
prompt = """
    I have data for some interviews where users were asked about their knowledge of
    and opinions on different home heating options. In the interview, users were asked about their knowledge
    of the Boiler Upgrade Scheme, a scheme that provides a subsidy to homeowners wishing to install a heatpump
    instead of getting a new gas boiler for their home.
    \n
    I have clustered user responses to the question {question}.
    \n
    One of the clusters contains the following user responses:
    {docs}
    The cluster is described by the following keywords: {keywords}
    \n
    Based on the information above, please provide a name and summary for the cluster as a JSON object with two fields: 
    - name: A short, informative name for the cluster 
    - description: A summary of views of users within the cluster. You can include sentiments they express, reasons for their views, their knowledge levels, and any other relevant information.
    \n
    Provide nothing except for this JSON dict.
    \n
    Example:
    {{
        "name": "Radiator upgrades",
        "description": "This cluster contains users who mention changing their radiators when questioned about home upgrades they would consider. Some express reluctance to alter the appearance of their home with larger radiators."
    }}
    """

parser = JsonOutputParser(pydantic_object=NameDescription)
    
final_prompt = PromptTemplate(
            template=prompt,
            input_variables=["question","docs", "keywords"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

model="llama3.2"

ollama_model = ChatOllama(
            model=model, temperature=0
            )

llm_chain = final_prompt | ollama_model | parser

def name_topics(topic_info: pd.DataFrame, llm_chain, topics, text_col='text_clean', top_words_col = 'Top Words',
                topic_label_col='Cluster', question='') -> dict:
    results = {}
        
    for topic in topics:
        logger.info(f"Processing topic {topic}")
        temp_df = topic_info[topic_info[topic_label_col] == topic]
        docs = temp_df[text_col].values[0]
        logger.info(f"Docs: {docs}")
        keywords = temp_df[top_words_col].values[0]
        logger.info(f"Keywords: {keywords}")
                
        try:
            output = llm_chain.invoke({ "question": question, "docs": docs, "keywords": keywords})
            print(output['name'], output['description'])
            results[topic] = output

        except Exception as e:
            logger.error(f"Error processing topic {topic}: {str(e)}")
            # errors.append({"topic": topic, "error": str(e)})
    
    return results

if __name__ == "__main__":
    data = pd.read_csv(PROJECT_DIR / "data/user_messages_min_len_9_w_sentiment.csv")
    
    data_dict = {}

    for q in data['q_number'].unique().tolist():
        df = data.copy()
        data_dict[q] = df[df['q_number'] == q]
        
    full_dfs = {}

    summary_dfs = {}

    for key in [0, 1,4,6,7,9]: # these questions are the ones with >100 data points in each
        temp_df = data_dict[key]
        
        question = temp_df['question'].unique()[0]
        
        temp_df['text_clean'] = temp_df['text_clean'].astype(str)
        docs = temp_df['text_clean'].tolist()
        embeddings = SENTENCE_MODEL.encode(docs, show_progress_bar=True)

        topics, probs = topic_model.fit_transform(docs, embeddings)
        
        rep_docs = topic_model.get_representative_docs()

        umap_2d = UMAP(random_state=RANDOM_SEED, n_components=2)
        embeddings_2d = umap_2d.fit_transform(embeddings)

        topic_lookup = topic_model.get_topic_info()[["Topic", "Name"]]

        temp_df['x'] = embeddings_2d[:,0]
        temp_df['y'] = embeddings_2d[:,1]
        temp_df['topic'] = topics
        temp_df['doc'] = docs
        df_vis = temp_df.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
        
        df_for_summarisation = pd.DataFrame(topic_model.get_topic_info())
        df_for_summarisation = df_for_summarisation[df_for_summarisation['Topic'] != -1]
        
        topic_list = df_for_summarisation['Topic'].to_list()
        
        results = name_topics(df_for_summarisation, llm_chain, topic_list, text_col='Representative_Docs', top_words_col = 'Representation',
                    topic_label_col='Topic', question=temp_df['question'].unique()[0])
        
        df_for_summarisation['name'] = df_for_summarisation['Topic'].map(lambda x: results[x]['name'] if x in results and isinstance(results[x], dict) and 'name' in results[x] else None)
        df_for_summarisation['description'] = df_for_summarisation['Topic'].map(lambda x: results[x]['description'] if x in results and isinstance(results[x], dict) and 'description' in results[x] else None)
        df_for_summarisation['question'] = question
        summary_dfs[key] = df_for_summarisation
        
        df_vis = df_vis.merge(df_for_summarisation[['Topic', 'name', 'description']], left_on='Topic', right_on='Topic', how='left')
        full_dfs[key] = df_vis
        
    # save two bar charts per question
    for key in full_dfs.keys():
        question = full_dfs[key]['question'].unique()[0]
            
        df = full_dfs[key]
            
        name_count_chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y('Name:N', title=None),
        x=alt.X('count()', title='Count')
        ).properties(
                #title='Count of Values of Name'
            )

        sentiment_proportion_chart = alt.Chart(df).mark_bar().encode(
                y=alt.Y('Name:N', title='Name'),
                x=alt.X('count()', stack='normalize', title='Proportion'),
                color=alt.Color('sentiment:N', title='Sentiment', scale=alt.Scale(domain=['Negative', 'Neutral', 'Positive'], range=['red', 'gray', 'green']))
            ).properties(
                # title='Proportion of Sentiment within Name'
            )

        # Combine the charts and add a title
        combined_chart = (name_count_chart | sentiment_proportion_chart).properties(
                title=f"{question}"
            )
        combined_chart.save(PROJECT_DIR / f"outputs/by_question/{question}_name_sentiment_chart.html")
    
    excel_file = PROJECT_DIR / 'outputs/by_question/question_topic_models.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # Iterate over unique values of 'question'
        for key in summary_dfs.keys():
            df = summary_dfs[key]
            df = df.explode('Representative_Docs')
            
            question = df['question'].unique()[0]
            
            # Save each summary table as a separate sheet in the Excel workbook
            sheet_name = question[:30]  # Excel sheet names are limited to 31 characters
            df[['question','Topic', 'Count', 'Name', 'Representation', 'name','description','Representative_Docs']].to_excel(writer, sheet_name=sheet_name, index=False)
        