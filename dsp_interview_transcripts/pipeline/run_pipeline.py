"""
Before running these scripts, you will need to make sure that Ollama is available.

You can do this by running the following from the terminal:
```
ollama serve
```
Alternatively check the bar at the top of your screen. A llama icon indicates that Ollama is running.
"""
import subprocess

from pathlib import Path

import plac


script_parent = Path(__file__).parent

from dsp_interview_transcripts import logger


@plac.annotations(
    production=("Run all scripts in production mode if True, otherwise in test mode", "flag", "production")
)
def main(production: bool = False):

    mode_arg = "-production" if production else ""

    # Run each script with the production argument
    logger.info("Running process_data.py...")
    subprocess.run(f"python {script_parent / 'process_data.py'} {mode_arg}", shell=True)
    logger.info("Running topic_modelling.py...")
    subprocess.run(f"python {script_parent / 'topic_modelling.py'} {mode_arg}", shell=True)
    logger.info("Running name_clusters.py...")
    subprocess.run(f"python {script_parent / 'name_clusters.py'} {mode_arg}", shell=True)
    logger.info("Running prep_output_tables.py...")
    subprocess.run(f"python {script_parent / 'prep_output_tables_and_figures.py'} {mode_arg}", shell=True)

    # # Top down approach
    # logger.info("Running top_down_analysis.py...")
    # subprocess.run(f"python {script_parent / 'top_down_analysis.py'}", shell=True)


if __name__ == "__main__":
    plac.call(main)
