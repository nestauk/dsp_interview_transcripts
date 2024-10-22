from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

import matplotlib.pyplot as plt
import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts.utils.data_cleaning import clean_data, convert_timestamp, remove_preamble, add_text_length

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BUFFER_SIZE = 1

chunker = SemanticChunker(HuggingFaceEmbeddings(model_name=MODEL), 
                              breakpoint_threshold_type="percentile", 
                              buffer_size=BUFFER_SIZE)

def chunk_with_uuids(pairs):
    # Turn every conversation into one big block of text (take the second item from every uuid-text tuple)
    text = '. '.join([t[1] for t in pairs])
    
    # Apply the semantic chunker to the text
    doc = Document(page_content=text)
    chunks = chunker.split_documents([doc])
    
    # For each chunk, find the corresponding UUIDs
    chunk_uuid_mapping = []
    
    for chunk in chunks:
        # For each chunk, identify the corresponding UUID(s)
        uuids_in_chunk = []
        chunk_text = chunk.page_content
        
        # Iterate through the original pairs to find corresponding UUIDs
        for uuid, text_segment in pairs:
            if text_segment in chunk_text:
                uuids_in_chunk.append(uuid)
        
        # Append to the mapping (chunk text -> corresponding UUIDs)
        chunk_uuid_mapping.append({
            'chunk_text': chunk_text,
            'uuids': uuids_in_chunk
        })
    
    return chunk_uuid_mapping

if __name__ == "__main__":
    # Read in the raw data
    data = pd.read_csv(PROJECT_DIR / 'data/qual_af_transcripts.csv')
    
    # Clean up the text a little and move audio transcriptions to the text column
    interviews_df = clean_data(data)
    
    # Make sure the conversations are sorted by time, so that the replies go in the right order
    interviews_df['timestamp_clean'] = interviews_df['timestamp'].apply(convert_timestamp)
    interviews_df = interviews_df.groupby('conversation', group_keys=False).apply(lambda x: x.sort_values('timestamp_clean'))
    
    # Remove everything up to when bot asks if the instructions are clear - everything before is just noise
    interviews_cleaned_df = interviews_df.groupby('conversation').apply(remove_preamble).reset_index(drop=True)
    
    df_grouped = interviews_cleaned_df.groupby('conversation').apply(lambda x: list(zip(x['uuid'], x['text_clean']))).reset_index(name='uuid_text_pairs')
    
    results = {}

    for idx, row in df_grouped.iterrows():
        conv = row['conversation']
        pairs = row['uuid_text_pairs']
        chunk_mapping = chunk_with_uuids(pairs)
        results[conv] = chunk_mapping
    
    flattened_data = []
    for conversation, chunks in results.items():
        for chunk in chunks:
            flattened_data.append({
                'conversation': conversation,
                'chunk_text': chunk['chunk_text'],
                'uuids': chunk['uuids']
            })

    df = pd.DataFrame(flattened_data)
    
    df = add_text_length(df, "chunk_text")
    # Remove chunks that are too short
    df = df[df['text_length'] > 4]
    
    df.to_csv(PROJECT_DIR / 'data/interviews_chunked.csv', index=False)
