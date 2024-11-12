
# Basic pipeline

1. Make sure you have a directory `data/` with the file `qual_af_transcripts.csv` in it.

2. Create a directory `outputs/` in the root of the repository.

3. Install [Ollama](https://ollama.com/) according to your operating system's instructions. Install [Llama3.2](https://ollama.com/library/llama3.2) (defaults to 3B) by running in your terminal `ollama pull llama3.2`.

4. Run `python dsp_interview_transcripts/pipeline/run_pipeline.py`. This runs the following scripts:
    - `process_data.py`. This will do some cleaning of the data (text cleaning, making sure the conversations are in order, concatenating consecutive messages by the same person within a conversation). It also applies a sentiment analysis model and also estimates which question from the interview guide is being discussed at each point.
    - `python dsp_interview_transcripts/pipeline/topic_modelling.py`. This runs a BERTopic style model, but the actual steps (dimensionality reduction, HDBSCAN, tfidf representation) are run separately
 because I wanted to normalise the embedding vectors before reducing the dimensionality.
    - `python dsp_interview_transcripts/pipeline/name_clusters.py`. This uses llama3.2 to generate a name and a description for each topic.
    - `python dsp_interview_transcripts/pipeline/prep_output_tables.py`. This formats the output data and also saves scatterplots locally.
    - `python dsp_interview_transcripts/pipeline/top_down_analysis.py`. This mimics the top-down/framework approach. It subsets the data according to which question is being discussed and then runs BERTopic on each subset.

The outputs are:

- `outputs/final/final_df.csv`: this is the main output file for the bottom-up approach, containing all user responses above a set length, some preceding context for each, the topic they're assigned to, the sentiment etc.
- `outputs/final/summary_info.csv`: this contains just the topic names and descriptions for the bottom-up approach, and some representative responses for each topic.
- `outputs/scatterplot... .html`: there are three scatterplots for the bottom-up approach: one coloured by topic, one coloured by question, and one coloured by sentiment.
- `outputs/by_question/`: this directory contains the output for the top-down approach: an info file on the topics generated for each question, and a scatterplot for each question.
