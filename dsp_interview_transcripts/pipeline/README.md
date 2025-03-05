
# Set up

* Make sure your raw data contains the following columns: ...

* Create a new directory in the S3 bucket `dsp-qualfml` with a short name for your project. Record this name as `project` in the config.

* Upload your raw data to `s3://dsp-qualfml/<your-project>/raw/`.

* Install [Ollama](https://ollama.com/) according to your operating system's instructions. Install [Llama3.2](https://ollama.com/library/llama3.2) (defaults to 3B) by running in your terminal `ollama pull llama3.2`. You can get ollama up and running by running `ollama serve` in the terminal.

* Populate `.env`

# Run an analysis!

## Testing

1. `python dsp_interview_transcripts/pipeline/process_data.py`: This will do some cleaning of the data (text cleaning, making sure the conversations are in order, concatenating consecutive messages by the same person within a conversation). It also applies a sentiment analysis model.

2. `python dsp_interview_transcripts/pipeline/topic_modelling.py -s <selection method> -c <min cluster size> -r <noise reduction strategy>`: This creates a topic model with BERTopic. You can supply some of the main hyperparameters (selection method, min cluster size, noise reduction strategy) as arguments to the script. If you don't supply them, the script will use the default values from the config. However, it's recommended to stay on this step for a bit and experiment until you find hyperparameters that produce reasonable clusters.

Some outputs that you can check in order to assess your topic model can be found in `dsp_interview_transcripts/outputs/<your-project>/`:

* `bertopic_topic_info_selection_<selection method>_min_length_9_min_cluster_<min cluster size>_red_<reduction strategy>.csv` - BERTopic summary info table

* `bertopic_visualization_selection_<selection method>_min_length_9_min_cluster_<min cluster size>_red_<reduction strategy>.html` - scatterplot showing the texts and topics in 2D vector space

* `noise_prop_selection_<selection method>_min_length_9_min_cluster_<min cluster size>_red_<reduction strategy>.txt` - a float indicating the proportion of data that ended up in the noise cluster

Once you're happy with the topic modelling results, record the hyperparameters that you want to use in `dsp_interview_transcripts/config/base.yaml`. These are used as the default arguments in `dsp_interview_transcripts/pipeline/process_data.py` so when you want to run everything in production mode, these hyperparameters will be taken directly from the config.

3. Create a summarisation prompt in `dsp_interview_transcripts/pipeline/prompts/`. Record the name of your prompt in `dsp_interview_transcripts/config/base.yaml`.

4. `python dsp_interview_transcripts/pipeline/name_clusters.py`. This uses llama3.2 to generate a name and a description for each topic. At this point you can iterate on the summarisation prompt (defined in the previous step).

5. `python dsp_interview_transcripts/pipeline/prep_output_tables.py`. This formats the output data and also saves scatterplots locally.

## Production

Once you are happy with the different components of the pipeline, you can run

```
python dsp_interview_transcripts/pipeline/run_pipeline.py -production
```
