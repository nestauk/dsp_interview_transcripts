"""
The script `dsp_interview_transcripts/pipeline/bit_france/analysis/topic_modelling.py` runs a topic model and saves, as one of its output,
a `.txt` file containing the proportion of texts that were assigned to the noise cluster.

This script allows you to iterate over these .txt files (assuming you have tried a few different hyperparameter combinations) and tells you which
one contains the smallest number i.e. which hyperparameter combination resulted in the least noise.

In practice this wasn't the most helpful metric.
"""

import os
import re

from dsp_interview_transcripts import PROJECT_DIR


def find_file_with_smallest_number(directory):
    """
    Finds the file ending in '.txt' within the given directory that contains the smallest float number.

    Parameters:
        directory (str): The path to the directory containing the '.txt' files.

    Returns:
        str: The name of the file containing the smallest float number, or None if no '.txt' files are found.
    """
    smallest_number = float("inf")
    smallest_file = None

    # Regex to match float numbers in the file content
    float_pattern = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as file:
                    content = file.read()

                # Find all float numbers in the file content
                numbers = [float(match) for match in float_pattern.findall(content)]

                # Check for the smallest number
                if numbers:
                    min_number = min(numbers)
                    if min_number < smallest_number:
                        smallest_number = min_number
                        smallest_file = filename

            except (ValueError, IOError) as e:
                print(f"Error processing file {filename}: {e}")

    return smallest_file


if __name__ == "__main__":

    directory_path = f"{PROJECT_DIR}/dsp_interview_transcripts/pipeline/bit_france/outputs"
    result = find_file_with_smallest_number(directory_path)
    if result:
        print(f"The file with the smallest number is: {result}")
    else:
        print("No '.txt' files with valid numbers were found in the directory.")
