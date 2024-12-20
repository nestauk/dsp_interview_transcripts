import os
import re

import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR


def process_text_files(input_dir, output_csv, profession):
    data = []  # List to store rows for the DataFrame

    # Iterate through all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)

            # Read the file content
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Use regex to extract speaker ID and their spoken text
            # Pattern 1: Speaker [\d]\n- Text
            pattern1 = re.findall(r"Speaker (\d)\n- (.*?)\n", content, re.DOTALL)

            # Pattern 2: Speaker [\d] [timestamp] - Text
            pattern2 = re.findall(r"Speaker (\d)\s+\d{2}:\d{2}\s+-\s+(.*?)\n", content, re.DOTALL)

            turns = pattern1 + pattern2

            # Add each turn as a row in the DataFrame
            for speaker_id, text in turns:
                data.append(
                    {
                        "speaker_id": int(speaker_id),
                        "text": text.strip(),
                        "file_name": file_name,
                        "profession": profession,
                    }
                )

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["speaker_id", "text", "file_name", "profession"])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Processed files saved to {output_csv}")


if __name__ == "__main__":

    input_dirs = [
        f"{PROJECT_DIR}/data/bit_france/converted/Salariés",
        f"{PROJECT_DIR}/data/bit_france/converted/Elus",
        f"{PROJECT_DIR}/data/bit_france/converted/Décideurs",
        f"{PROJECT_DIR}/data/bit_france/converted/Médecins du travail",
    ]

    for input_directory in input_dirs:
        profession = input_directory.split("/")[-1]
        output_directory = PROJECT_DIR / f"data/bit_france/converted/{profession}/output.csv"

        process_text_files(input_directory, output_directory, profession)

    dataframes = []

    for directory in input_dirs:
        file_path = os.path.join(directory, "output.csv")
        if os.path.exists(file_path):  # Ensure the file exists
            df = pd.read_csv(file_path)
            dataframes.append(df)
        else:
            print(f"File not found: {file_path}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path = f"{PROJECT_DIR}/data/bit_france/converted/combined_transcripts.csv"
    combined_df.to_csv(output_path, index=False)

    print(f"Combined CSV saved to {output_path}")
