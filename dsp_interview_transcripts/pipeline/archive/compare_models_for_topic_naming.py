import ast
from langchain_community.chat_models import ChatOllama
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

from dsp_interview_transcripts import logger, PROJECT_DIR

class NameDescription(BaseModel):
    name: str = Field(description="Informative name for this group of documents")
    description: str = Field(description="Description of this group of documents")
    
prompt = """
    I have performed text clustering on some interviews where users were asked about their knowledge of
    and opinions on different home heating options.
    \n
    One of the clusters contains the following user responses:
    {docs}
    The cluster is described by the following keywords: {keywords}
    \n
    Based on the information above, please provide a name and description for the cluster as a JSON object with two fields: 
    - name (a short, informative name for the cluster) 
    - description (a brief description of the cluster).
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

if __name__ == "__main__":
    topic_info = pd.read_csv(PROJECT_DIR / "outputs/user_messages_cluster_size_10_topic_info.csv")

    for col in ['Representation', 'KeyBERT', 'MMR', 'Representative_Docs']:
        topic_info[col] = topic_info[col].apply(ast.literal_eval)
        
    
    topics = topic_info["Topic"].unique()

    
    for model in ["llama3.1", "llama3.2", "phi3"]:
        
        logger.info(f"Obtaining names and descriptions with model: {model}")
        
        ollama_model = ChatOllama(
            model=model, temperature=0
            )
        llm_chain = final_prompt | ollama_model | parser
        
        results = {}
        
        for topic in topics:
            if topic==-1:
                continue
            else:
                logger.info(f"Processing topic {topic} with model {model}")
                temp_df = topic_info[topic_info["Topic"] == topic]
                docs = temp_df['Representative_Docs'].values[0]
                keywords = temp_df['MMR'].values[0]
                output = llm_chain.invoke({"docs":docs, "keywords":keywords})
                results[topic] = output
            
        topic_info[f'{model}_name'] = topic_info['Topic'].map(lambda x: results[x]['name'] if x in results else None)
        topic_info[f'{model}_description'] = topic_info['Topic'].map(lambda x: results[x]['description'] if x in results else None)
    
    logger.info("Savng output...")
    topic_info.to_csv(PROJECT_DIR / "outputs/user_messages_cluster_size_10_topic_info_with_names_descriptions.csv", index=False)
    logger.info("Done!")
        
    
