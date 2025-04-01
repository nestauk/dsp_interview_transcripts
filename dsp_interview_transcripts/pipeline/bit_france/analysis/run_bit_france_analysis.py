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


def main():

    logger.info("Running convert_txt.py...")
    subprocess.run(f"python {script_parent / 'convert_txt.py'}", shell=True)

    logger.info("Running convert_txt_df.py...")
    subprocess.run(f"python {script_parent / 'convert_txt_df.py'}", shell=True)

    logger.info("Running assign_roles.py...")
    subprocess.run(f"python {script_parent / 'assign_roles.py'}", shell=True)

    logger.info("Running identify_maquettes_section.py...")
    subprocess.run(f"python {script_parent / 'identify_maquettes_section.py'}", shell=True)

    logger.info("Running add_context.py...")
    subprocess.run(f"python {script_parent / 'add_context.py'}", shell=True)

    logger.info("Running topic_modelling.py...")
    subprocess.run(f"python {script_parent / 'topic_modelling.py'} -s leaf -l 9 -c 15 -r probabilities", shell=True)

    logger.info("Running prep_outputs.py...")
    subprocess.run(f"python {script_parent / 'prep_outputs.py'}", shell=True)

    logger.info("Running name_clusters.py...")
    subprocess.run(f"python {script_parent / 'name_clusters.py'}", shell=True)

    logger.info("Running prep_outputs_for_report.py...")
    subprocess.run(f"python {script_parent / 'prep_outputs_for_report.py'}", shell=True)


if __name__ == "__main__":
    plac.call(main)
