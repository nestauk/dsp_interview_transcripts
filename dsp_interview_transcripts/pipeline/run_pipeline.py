"""
Before running these scripts, you will need to make sure that Ollama is available.

You can do this by running the following from the terminal:
```
ollama serve
```
Alternatively check the bar at the top of your screen. A llama icon indicates that Ollama is running.
"""
from pathlib import Path
import subprocess
import os

script_parent = Path(__file__).parent

# All of these scripts cover topic modelling across *all* responses. The top down analysis is separate
subprocess.run(f"python {script_parent / 'process_data.py'}", shell=True)
subprocess.run(f"python {script_parent / 'topic_modelling.py'}", shell=True)
subprocess.run(f"python {script_parent / 'name_clusters.py'}", shell=True)
subprocess.run(f"python {script_parent / 'prep_output_tables.py'}", shell=True)

# Top down approach
subprocess.run(f"python {script_parent / 'top_down_analysis.py'}", shell=True)