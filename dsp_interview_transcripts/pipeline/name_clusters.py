import ast
import glob
import json
from langchain_community.chat_models import ChatOllama
import os
import numpy as np
import pandas as pd
import re
from typing import List

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from dsp_interview_transcripts import logger, PROJECT_DIR

class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")
    
prompt = """
    I have performed text clustering on some interviews where users were asked about their knowledge of
    and opinions on different home heating options. In the interview, users were asked about their knowledge
    of the Boiler Upgrade Scheme, a scheme that provides a subsidy to homeowners wishing to install a heatpump
    instead of getting a new gas boiler for their home.
    \n
    One of the clusters contains the following user responses from the interviews:
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
        "name": "Energy Efficiency",
        "description": "This cluster contains users responses about energy efficiency when choosing home heating options. The users have varying degrees of knowledge about the efficiency of different systems. Some reasons for wanting to improve efficiency include environmental concerns and cost concerns."
    }}
    """

parser = JsonOutputParser(pydantic_object=NameDescription)
    
final_prompt = PromptTemplate(
            template=prompt,
            input_variables=["docs", "keywords"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

model="llama3.2"

ollama_model = ChatOllama(
            model=model, temperature=0
            )

llm_chain = final_prompt | ollama_model | parser

INPUT_PATH = PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv"

def save_list_of_dicts_as_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for entry in data:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")
            
def name_topics(topic_info: pd.DataFrame, llm_chain, topics: List[str], text_col='text_clean', top_words_col = 'Top Words',
                topic_label_col='Cluster') -> dict:
    results = {}
        
    for topic in topics:
        logger.info(f"Processing topic {topic}")
        temp_df = topic_info[topic_info[topic_label_col] == topic]
        docs = temp_df[text_col].values[0]
        logger.info(f"Docs: {docs}")
        keywords = temp_df[top_words_col].values[0]
        logger.info(f"Keywords: {keywords}")
                
        try:
            output = llm_chain.invoke({"docs": docs, "keywords": keywords})
            print(output['name'], output['description'])
            results[topic] = output

        except Exception as e:
            logger.error(f"Error processing topic {topic}: {str(e)}")
            errors.append({"topic": topic, "error": str(e)})
    
    return results

if __name__ == "__main__":

    errors = []
    
    topic_info = pd.read_csv(INPUT_PATH)
    
    topic_info = topic_info.groupby(['Cluster', 'Top Words'])['text_clean'].apply(list).reset_index()
    topic_info['Cluster'] = topic_info['Cluster'].astype(str)
                
    topics = topic_info["Cluster"].unique().tolist()
        
    results = name_topics(topic_info, llm_chain, topics, text_col='text_clean', top_words_col = 'Top Words',
                topic_label_col='Cluster')
        
    ## Some complicated conditionals to check that what's in `results` can be parsed ##            
    topic_info[f'{model}_name'] = topic_info['Cluster'].map(lambda x: results[x]['name'] if x in results and isinstance(results[x], dict) and 'name' in results[x] else None)
    topic_info[f'{model}_description'] = topic_info['Cluster'].map(lambda x: results[x]['description'] if x in results and isinstance(results[x], dict) and 'description' in results[x] else None)
            
    logger.info("Saving output...")
    topic_info.to_csv(PROJECT_DIR / f"outputs/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv", index=False)
    logger.info("Done!")
    
    ## Save errors so that they can be loaded in and manually reprocessed if necessary ##
    save_list_of_dicts_as_jsonl(errors, PROJECT_DIR/"outputs/interim/errors.jsonl")
