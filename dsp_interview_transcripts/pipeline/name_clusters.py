import ast
import glob
import json
from langchain_community.chat_models import ChatOllama
import os
import pandas as pd
import re

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from dsp_interview_transcripts import logger, PROJECT_DIR

class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")
    
prompt = """
    I have performed text clustering on some interviews where users were asked about their knowledge of
    and opinions on different home heating options. These texts may contain user responses, interviewer questions,
    or both.
    \n
    One of the clusters contains the following texts from the interviews:
    {docs}
    The cluster is described by the following keywords: {keywords}
    \n
    Based on the information above, please provide a name and description for the cluster as a JSON object with two fields: 
    - name (a short, informative name for the cluster) 
    - description (a brief description of the cluster).
    \n
    Provide nothing except for this JSON dict.
    \n
    Example:
    {{
        "name": "Energy Efficiency",
        "description": "This cluster focuses on users' concerns about energy efficiency when choosing home heating options."
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

INPUT_PATH = PROJECT_DIR / "outputs"


def save_list_of_dicts_as_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for entry in data:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")

if __name__ == "__main__":

    files = glob.glob(os.path.join(INPUT_PATH, '**', '*topic_info.csv'), recursive=True)

    filelist = []

    for file in files:
        filelist.append(os.path.basename(file))
        
    output = []

    for file in filelist:
        split1 = re.split('_cluster_size', file)
        source = split1[0]
        split2 = re.split('_', split1[1])
        cluster_size = split2[1]
        output.append({'filename': file, 'source': source, 'cluster_size': cluster_size})
        
    files_df = pd.DataFrame(output)
    
    errors = []
    
    for file in filelist:
        topic_info = pd.read_csv(PROJECT_DIR / f"outputs/{file}")
        source = files_df[files_df['filename']==file]['source'].values[0]
        cluster_size = files_df[files_df['filename']==file]['cluster_size'].values[0]
        print(f"Processing {source} with cluster size {cluster_size}")

        for col in ['Representation', 'KeyBERT', 'MMR', 'Representative_Docs']:
            topic_info[col] = topic_info[col].apply(ast.literal_eval)
                
        topics = topic_info["Topic"].unique()
        
        results = {}
        
        for topic in topics:
            if topic==-1:
                continue
            else:
                logger.info(f"Processing topic {topic} with model {model}")
                temp_df = topic_info[topic_info["Topic"] == topic]
                docs = temp_df['Representative_Docs'].values[0]
                keywords = temp_df['MMR'].values[0]
                
                try:
                    output = llm_chain.invoke({"docs": docs, "keywords": keywords})
                    results[topic] = output

                except Exception as e:
                    logger.error(f"Error processing topic {topic}: {str(e)}")
                    errors.append({"file": file, "source": source, "cluster_size": cluster_size, "topic": topic, "error": str(e)})    
        
        ## Some complicated conditionals to check that what's in `results` can be parsed ##            
        topic_info[f'{model}_name'] = topic_info['Topic'].map(lambda x: results[x]['name'] if x in results and isinstance(results[x], dict) and 'name' in results[x] else None)
        topic_info[f'{model}_description'] = topic_info['Topic'].map(lambda x: results[x]['description'] if x in results and isinstance(results[x], dict) and 'description' in results[x] else None)
            
        logger.info("Saving output...")
        topic_info.to_csv(PROJECT_DIR / f"outputs/{source}_cluster_size_{cluster_size}_topic_info_with_names_descriptions.csv", index=False)
        logger.info("Done!")
    
    ## Save errors so that they can be loaded in and manually reprocessed if necessary ##
    save_list_of_dicts_as_jsonl(errors, PROJECT_DIR/"outputs/interim/errors.jsonl")