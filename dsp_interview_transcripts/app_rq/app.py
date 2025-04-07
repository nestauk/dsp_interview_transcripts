"""
Navigate to folder and then run:
```
streamlit run app.py
```
"""
import uuid

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.pipeline import build_question_prompt_dict
from utils.pipeline import convert_transcripts_df_to_dict
from utils.pipeline import normalize_uuid
from utils.pipeline import run_batch_check
from utils.summarize import summarize_and_quote


PROMPT_PATH = Path("prompts/llm_check_system_a.txt")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

st.title("Research Question Explorer")
st.markdown("Upload conversation data and define research questions to explore with LLMs.")

# Upload data
data_file = st.file_uploader(
    "Upload a CSV file with columns: conversation, role, text (and optionally uuid)", type="csv"
)
data = None
if data_file:
    data = pd.read_csv(data_file)

    st.success("Data uploaded successfully!")

    st.markdown("### Step 1: Select relevant columns")

    cols = data.columns.tolist()

    conv_id_col = st.selectbox("Select the conversation ID column", cols)
    role_col = st.selectbox("Select the speaker role column", cols)
    text_col = st.selectbox("Select the text column", cols)
    uuid_col = st.selectbox("Select the unique text ID column (optional)", ["None"] + cols)

    if uuid_col == "None":
        data["uuid"] = [str(uuid.uuid4()) for _ in range(len(data))]
        uuid_col = "uuid"
    data[uuid_col] = data[uuid_col].apply(normalize_uuid)

# Research questions input
rq_text = st.text_area("Enter research questions, one per line")
if rq_text:
    research_questions = rq_text.strip().splitlines()
    rq_dict = {f"rq_{i+1}": q for i, q in enumerate(research_questions)}
else:
    rq_dict = {}

if st.button("Run Analysis") and data is not None and rq_dict:
    with st.spinner("Running LLM processing..."):
        prompt_template = PROMPT_PATH.read_text()
        conversation_dict = convert_transcripts_df_to_dict(data, conv_id_col, role_col, text_col, uuid_col)
        prompt_dict = build_question_prompt_dict(rq_dict, prompt_template)
        output_paths = run_batch_check(conversation_dict, prompt_dict, OUTPUT_DIR)

    st.success("LLM processing complete! Generating summaries and quotes...")

    for rq_id, question in rq_dict.items():
        st.subheader(f"RQ: {question}")
        path = Path(output_paths[rq_id])
        if path.exists():
            df = pd.read_json(path, lines=True)
            # Use all extracted text fields
            extracted_texts = [txt for sublist in df["text"] for txt in sublist]
            answer, quotes = summarize_and_quote(extracted_texts, question)

            st.markdown(f"**Summary Answer:** {answer}")
            st.markdown("**Key Quotes:**")
            for q in quotes:
                st.markdown(f"> {q}")
        else:
            st.warning(f"No output found for {question}")
