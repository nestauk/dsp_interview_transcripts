
# Set up

## Data
Download data and store it in `dsp_interview_transcripts/app/data/`

The files you need are:

* [s3://dsp-qualfml/raw/qual_af_transcripts_cleaned.csv](https://eu-west-2.console.aws.amazon.com/s3/object/dsp-qualfml?region=eu-west-2&bucketType=general&prefix=raw/qual_af_transcripts_cleaned.csv)

* [s3://dsp-qualfml/interim/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv](https://eu-west-2.console.aws.amazon.com/s3/object/dsp-qualfml?region=eu-west-2&bucketType=general&prefix=interim/user_messages_min_len_9_w_sentiment_topics_representative_docs.csv)

* [s3://dsp-qualfml/interim/user_messages_min_len_9_w_sentiment_topics.csv](https://eu-west-2.console.aws.amazon.com/s3/object/dsp-qualfml?region=eu-west-2&bucketType=general&prefix=interim/user_messages_min_len_9_w_sentiment_topics.csv)

* [s3://dsp-qualfml/interim/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv](https://eu-west-2.console.aws.amazon.com/s3/object/dsp-qualfml?region=eu-west-2&bucketType=general&prefix=interim/user_messages_min_len_9_w_sentiment_topics_with_names_descriptions.csv)

If we continue the app long-term, we will set up a service account to read data from s3 directly but for now you need to download and store the data locally.

## Virtual environment
Create a virtual env for the app separate from the general project environment:
```
python3.10 -m venv env
source env/bin/activate
pip install -r dsp_interview_transcripts/app/requirements.txt
```
The reason for this is that the overall project has some hefty dependencies e.g. sentence transformers, pytorch and so on. We do not need these for the app so avoiding packages that download language models means we can get away with using a smaller EC2 instance for the app.

Long term, it may be better practice to have the app in its own repo so that front-end development is entirely separate from the data processing/ML aspects of the project.

# How to run the app locally

* Activate the virtual env (see above)

* `cd dsp_interview_transcripts/app/`

* `python app_pages.py`

You will see from the message in your terminal that the app is running on a url, e.g. `http://127.0.0.1:8050`. Open this in your browser to see the app.

# Deployment

The EC2 instance we use is called `qualfml` and it is a `t3.small` instance.

## Updating app data

Follow the instructions above to download the necessary data files.

Once you've downloaded these files from s3 to your own computer, follow the instructions from [slide 13 of Jack's have-a-go](https://docs.google.com/presentation/d/1Pu315-w6TPWhJMUojv2oOMWyEBbVy99pq1mVEXh3FCI/edit#slide=id.g31031850c6d_0_30) to copy those files to the instance.

```
scp -i <location-of-pem.pem> <data-to-copy.csv> ubuntu@<public-ipv4-dns>:/home/ubuntu/dsp_interview_transcripts/app/data/
```
