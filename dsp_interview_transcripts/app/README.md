
# Set up

## Data
Download data and store it in `dsp_interview_transcripts/app/data/`

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
