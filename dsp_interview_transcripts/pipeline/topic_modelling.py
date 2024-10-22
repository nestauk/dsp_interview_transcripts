import numpy as np
import pandas as pd
import random
from sentence_transformers import SentenceTransformer
from collections import defaultdict

from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from dsp_interview_transcripts import PROJECT_DIR, logger

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# PyTorch seed (used by SentenceTransformer)
import torch

torch.manual_seed(RANDOM_SEED)

SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DATA_PATH = PROJECT_DIR / "data/user_messages_min_len_9_w_sentiment.csv"

MIN_CLUSTER_SIZE = 20

umap_model = UMAP(
            n_neighbors=15,
            n_components=50,
            min_dist=0.1,
            metric="cosine",
            random_state=RANDOM_SEED,
            )

hdbscan_model = HDBSCAN(
                min_cluster_size=20,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

vectorizer_model = TfidfVectorizer(
                stop_words="english",
                min_df=1,
                max_df=0.85,
                ngram_range=(1, 3),
            )

def stratified_sample(group, n=10):
    # Calculate the sample size for each combination of 'question' and 'sentiment'
    stratified_sample = group.groupby(['question', 'sentiment']).apply(lambda x: x.sample(frac=min(1, n / len(group)), random_state=42))
    
    # Reset the index to tidy up the resulting DataFrame
    return stratified_sample.reset_index(drop=True)

if __name__ == "__main__":
    user_messages = pd.read_csv(DATA_PATH)
    
    docs = user_messages['text_clean'].tolist()
    embeddings = SENTENCE_MODEL.encode(docs, show_progress_bar=True)

    umap_vectors = umap_model.fit_transform(embeddings)

    umap_df = pd.DataFrame(umap_vectors)
    umap_df.columns = umap_df.columns.map(str)
    user_messages_w_umap = pd.concat([user_messages.reset_index(), umap_df], axis=1)

    umap_vars = [f'{i}' for i in range(50)]
    model_vars = umap_vars

    # Normalise the umap vectors
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(user_messages_w_umap[model_vars])
    
    clusters = hdbscan_model.fit_predict(df_normalized)
    cluster_probabilities = hdbscan_model.probabilities_
    user_messages_w_umap['label'] = clusters
    user_messages_w_umap['probs'] = cluster_probabilities
    logger.info(user_messages_w_umap['label'].value_counts())
    
    # 2d embeddings for visualisation
    umap_2d = UMAP(random_state=RANDOM_SEED, n_components=2)
    embeddings_2d = umap_2d.fit_transform(embeddings)
    
    ### topic representations ###
    cluster_groups = user_messages_w_umap.groupby('label').agg({'text_clean': ' '.join}).reset_index()

    tfidf_matrix = vectorizer_model.fit_transform(cluster_groups['text_clean'].to_list())

    feature_names = vectorizer_model.get_feature_names_out()

    # Create a dictionary to hold top words for each cluster
    top_words_per_cluster = defaultdict(list)

    # Number of top words you want to display per cluster
    n_top_words = 10

    # Iterate over each cluster and get top words
    for cluster_idx, tfidf_scores in enumerate(tfidf_matrix):
        # Get indices of top n words within the cluster
        top_word_indices = tfidf_scores.toarray()[0].argsort()[:-n_top_words - 1:-1]
        
        # Get the top words corresponding to the top indices
        top_words = [feature_names[i] for i in top_word_indices]
        
        # Append the words to the dictionary
        top_words_per_cluster[cluster_groups.iloc[cluster_idx]['label']] = ', '.join(top_words)
        
    top_words_df = pd.DataFrame(list(top_words_per_cluster.items()), columns=['Cluster', 'Top Words'])
    
    user_messages_w_umap_topics = pd.merge(user_messages_w_umap, top_words_df, left_on='label', right_on='Cluster', how='left')
    user_messages_w_umap_topics['x'] = embeddings_2d[:, 0]
    user_messages_w_umap_topics['y'] = embeddings_2d[:, 1]
    
    user_messages_w_umap_topics.to_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics.csv", index=False)
    
    
    ### save most representative documents ###
    
    filtered_df = user_messages_w_umap_topics[(user_messages_w_umap_topics['probs'] >= 0.5) & (user_messages_w_umap_topics['Cluster'] != -1)]
    
    # Group by 'Cluster' and apply the stratified sampling
    sampled_texts = filtered_df.groupby('Cluster').apply(stratified_sample).reset_index(drop=True)

    # Keep only the first 10 samples per cluster
    sampled_texts = sampled_texts.groupby('Cluster').head(10)
    
    sampled_texts[['Cluster', 'Top Words', 'text_clean', 'sentiment', 'question', 'context','conversation', 'uuid', 'probs']].to_csv(PROJECT_DIR / "outputs/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv", index=False)