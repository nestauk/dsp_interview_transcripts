

# Set up

* Download [this file](https://docs.google.com/spreadsheets/d/10SQ8S3ryKu9-zs0d-7WQ0_Kze9Ef6OAt-k79rK9s4EE/edit?gid=1143506821#gid=1143506821) and store it as `data/bit_france/converted/file_mapping.csv`

* The transcripts for each professional group (Salariés, Médicins du travail, Elus and Décideurs) can be found in subfolders [here](https://drive.google.com/drive/folders/1CAHxIzTJLMmwRQb-1bZi4ezDXaKsDeXI). Download each of these folders and store them in `data/bit_france/`. You will have the following folders:

```
data/bit_france/Décideurs/
data/bit_france/Elus/
data/bit_france/Médicins du travail/
data/bit_france/Salariés/
```

Download [this file](https://docs.google.com/spreadsheets/d/1XJAxPFh_n8UIAC-wtBaE1wPVUCbaDguHdZCfQedAIeE/edit?gid=1062484915#gid=1062484915) and store it as `data/bit_france/converted/bit_france_interview_sections - ALT_data_labelled_with_new_sections.csv`. This file has been **manually labelled** to indicate where the section of the interview about interventions/prototypes begins. See the notebook `dsp_interview_transcripts/pipeline/bit_france/analysis/identify_interview_sections.ipynb` for more information.

# Analysis

**Steps 1-9 below can be triggered by running `dsp_interview_transcripts/pipeline/bit_france/analysis/run_bit_france_analysis.py`**. All the inputs must be downloaded before running this script; all outputs get saved locally.

## Data transformations

1. `dsp_interview_transcripts/pipeline/bit_france/analysis/convert_txt.py` converts the `.docx` files to `.txt` files

2. `dsp_interview_transcripts/pipeline/bit_france/analysis/convert_txt_df.py` concatenates the `.txt` files into one dataframe

3. `dsp_interview_transcripts/pipeline/bit_france/analysis/assign_roles.py` uses a mapping table to determine within each interview, who is the interviewer and who is the informant

At this point, the manually labelled file `data/bit_france/converted/bit_france_interview_sections - ALT_data_labelled_with_new_sections.csv` is introduced into the pipeline.

4. `dsp_interview_transcripts/pipeline/bit_france/analysis/identify_maquettes_section.py` filters the dataframe to only include the section of the interview about interventions/prototypes.

5. `dsp_interview_transcripts/pipeline/bit_france/analysis/add_context.py` creates overlapping windows of text. Each chunk of text is of the format
```
INTERVIEWER: <text>
INFORMANT: <text>
INTERVIEWER: <text>
INFORMANT: <text>
```

## Topic modelling and summarisation

The scripts continue in this order:

6. `dsp_interview_transcripts/pipeline/bit_france/analysis/topic_modelling.py` creates a topic model for each professional group.

Note that the script takes different hyperparameter values as input. The final values used are:

* selection method: leaf

* minimum text length: 9 (however this doesn't make a difference as the way we have chunked the text means we don't have any super short texts)

* minimum cluster size: 15

* noise reduction strategy: probabilities

(These should be moved to a config file but this hasn't happened yet)

7. `dsp_interview_transcripts/pipeline/bit_france/analysis/prep_outputs.py` extracts representative documents and prepares a dataframe for visualisation with 2D embeddings.

8. `dsp_interview_transcripts/pipeline/bit_france/analysis/name_clusters.py` creates summaries for each cluster with llama3.2.

9. `dsp_interview_transcripts/pipeline/bit_france/analysis/prep_outputs_for_report.py` prepares the final outputs for the report.

## Report

Once you have run all of the processes above, you can access the final report in `dsp_interview_transcripts/pipeline/bit_france/report/bit_france_report.qmd`
