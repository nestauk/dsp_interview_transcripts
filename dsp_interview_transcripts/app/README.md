🌱 This directory contains the newer version of the app. This version allows the user to upload their own data and run an analysis.

# How to run the app locally

```
python dsp_interview_transcripts/app/app_tabs.py
```

You will see from the message in your terminal that the app is running on a url, e.g. `http://127.0.0.1:8050`. Open this in your browser to see the app.

# Structure

`app_tabs.py`: This controls the overall layout of the app. There are also variables stored silently in the layout (`dcc.Store(...)`) that are used to store data between callbacks and between tabs.

Each page of the app has one script controlling its layout in `layout/`, and a separate one containing its backend logic in `callbacks/`.

Styling is controlled by two files:

* `style.py`

* `assets/style.css` - this contains some custom css classes for the app.

# Secrets needed

```
LLM_SERVICE=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_OPENAI_API_VERSION=...
```

# Deploying on EC2

## Set up

* Create a new EC2 instance. We are currently using `t3.medium`.

* SSH into the instance and install the following (from [Jack's EC2 instructions](https://docs.google.com/presentation/d/1Pu315-w6TPWhJMUojv2oOMWyEBbVy99pq1mVEXh3FCI/edit?slide=id.g31031850c6d_0_25#slide=id.g31031850c6d_0_25)):
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install unzip
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install build-essential
sudo apt-get install manpages-dev
```

* Install poetry

* Clone this repo

* Install just the app-specific dependencies: `poetry install --only app`

## Run the app

* Run the app with `nohup python dsp_interview_transcripts/app/app_tabs.py > app.log 2>&1 &`

## Access
Currently this app is available to anyone on the Nesta VPN.

## Storing user data

The app stores user data in `dsp_interview_transcripts/outputs/`. The `uuid` library is used in the app to generate a unique identifier for each user, which is used to create a folder for each user in `dsp_interview_transcripts/outputs/`.

There is a cron job defined in `dsp_interview_transcripts/automation/` that runs every day at 00:00 and deletes all folders in `dsp_interview_transcripts/outputs/`. It is currently live on the EC2 instance and it was set up by:

* Running `crontab -e`

* Adding the following line to the crontab file: `0 0 * * * /home/ubuntu/dsp_interview_transcripts/automation/cleanup_outputs.sh`
