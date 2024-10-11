from pathlib import Path
import subprocess
import os

script_parent = Path(__file__).parent

subprocess.run(f"python {script_parent / 'process_data.py'}", shell=True)
subprocess.run(f"python {script_parent / 'chunk_interviews.py'}", shell=True)
subprocess.run(f"python {script_parent / 'segment_interviews.py'}", shell=True)
subprocess.run(f"python {script_parent / 'topic_modelling.py'}", shell=True)
subprocess.run(f"python {script_parent / 'tname_clusters.py'}", shell=True)
