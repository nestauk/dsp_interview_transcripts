import os

from pathlib import Path

from docx import Document

from dsp_interview_transcripts import PROJECT_DIR


def convert_docx_to_txt(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".docx"):
            input_path = os.path.join(input_dir, file_name)
            output_file_name = os.path.splitext(file_name)[0] + ".txt"
            output_path = os.path.join(output_dir, output_file_name)

            try:
                doc = Document(input_path)
                with open(output_path, "w", encoding="utf-8") as txt_file:
                    for paragraph in doc.paragraphs:
                        txt_file.write(paragraph.text + "\n")
                print(f"Converted: {file_name} -> {output_file_name}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")


if __name__ == "__main__":

    for input_directory in [
        f"{PROJECT_DIR}/data/bit_france/Salariés",
        f"{PROJECT_DIR}/data/bit_france/Elus",
        f"{PROJECT_DIR}/data/bit_france/Décideurs",
        f"{PROJECT_DIR}/data/bit_france/Médecins du travail",
    ]:
        profession = input_directory.split("/")[-1]
        output_directory = PROJECT_DIR / f"data/bit_france/converted/{profession}"
        convert_docx_to_txt(input_directory, output_directory)
