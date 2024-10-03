
The pipeline is very simple at this stage.

1. Make sure you have a directory `data/` with the file `qual_af_transcripts.csv` in it.

2. Run `python dsp_interview_transcripts/pipeline/run_pipeline.py`. This runs the following scripts:
    - `process_data.py`. This will do some cleaning of the data (text cleaning, making sure the conversations are in order, concatenating consecutive messages by the same person within a conversation) and save two outputs: a file with just user messages over a certain length; one with the user messages, plus the immediately preceding bot message (in order to see if this extra context helps the text clustering and topic representations).
    - `python dsp_interview_transcripts/pipeline/segment_interviews.py`. This uses some basic comparison between the topic guide and the BOT messages to figure out which parts of the interview might correspond to which question from the topic guide. A separate dataset is saved for each of the questions from the topic guide.
    - `python dsp_interview_transcripts/pipeline/topic_modelling.py`. This runs BERTopic on the datasets provided above, and also iterates over a few different minimum cluster sizes.

3. Run `streamlit run dsp_interview_transcripts/pipeline/app.py`. This launches a streamlit app that allows you to inspect the outputs of BERTopic that we created in the step above.