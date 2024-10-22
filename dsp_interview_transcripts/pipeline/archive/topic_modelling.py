from bertopic import BERTopic
import numpy as np
import pandas as pd
import random
from sentence_transformers import SentenceTransformer

from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import (
    KeyBERTInspired,
    MaximalMarginalRelevance,
    # OpenAI,
    PartOfSpeech,
)

from dsp_interview_transcripts import PROJECT_DIR

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# PyTorch seed (used by SentenceTransformer)
import torch

torch.manual_seed(RANDOM_SEED)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INTERVIEW_SECTIONS = [f"interview_q_{i}" for i in range(-1, 10) if i != 4] # there were no hits for q4 for some reason
DATA_SOURCES = ["user_messages",
                "q_and_a",
                "interviews_chunked"
                ] + INTERVIEW_SECTIONS

MIN_CLUSTER_SIZES = [5, 10, 20, 50, 100]

if __name__ == "__main__":
    
    for source in DATA_SOURCES:
        for cluster_size in MIN_CLUSTER_SIZES:
            if source in INTERVIEW_SECTIONS and cluster_size>10:
                continue
            if source=="interviews_chunked" and cluster_size>20:
                continue
            
            df = pd.read_csv(PROJECT_DIR / f"data/{source}.csv")
            
            if source=="user_messages" or source in INTERVIEW_SECTIONS:
                text_col = "text_clean"
            elif source=="interviews_chunked":
                text_col = "chunk_text"
            else:
                text_col = "q_and_a"
            
            umap_model = UMAP(
            n_neighbors=15,
            n_components=50,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_SEED,
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=cluster_size,
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
                "MMR": mmr_model,
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
            
            # Convert NaNs to empty strings
            df[text_col] = df[text_col].astype(str)
            docs = df[text_col].tolist()
            embeddings = SENTENCE_MODEL.encode(docs, show_progress_bar=True)

            topics, probs = topic_model.fit_transform(docs, embeddings)
            
            # save topic model
            topic_model.save(
            PROJECT_DIR / f"outputs/topic_model_{source}_cluster_size_{cluster_size}",
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=SENTENCE_MODEL,
            )

            rep_docs = topic_model.get_representative_docs()
            
            # Save representative docs and keywords
            # (These will be used to obtain summaries)
            topic_info = pd.DataFrame(topic_model.get_topic_info())
            topic_info.to_csv(PROJECT_DIR / f"outputs/{source}_cluster_size_{cluster_size}_topic_info.csv", index=False)

            umap_2d = UMAP(random_state=RANDOM_SEED, n_components=2)
            embeddings_2d = umap_2d.fit_transform(embeddings)

            topic_lookup = topic_model.get_topic_info()[["Topic", "Name"]]

            df_vis = pd.DataFrame(embeddings_2d, columns=["x", "y"])
            df_vis["topic"] = topics
            df_vis = df_vis.merge(topic_lookup, left_on="topic", right_on="Topic", how="left")
            df_vis["doc"] = docs

            if source=="interviews_chunked":
                base_df = df[['conversation',text_col]]
            else:
                base_df = df[["uuid","conversation",text_col]]
                
            df_vis = pd.merge(
                    base_df,
                    df_vis,
                    left_on=text_col,
                    right_on="doc",
                    how="outer",
                )
            
            df_vis.to_csv(PROJECT_DIR / f"outputs/{source}_cluster_size_{cluster_size}_vis.csv", index=False)
