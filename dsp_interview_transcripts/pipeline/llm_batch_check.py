import asyncio
import os
import time

from functools import reduce
from pathlib import Path

import pandas as pd
import plac

from discovery_utils.utils.llm import batch_check

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import config
from dsp_interview_transcripts import logging
from dsp_interview_transcripts.getters.raw import get_raw_transcripts_cleaned


PROMPT_PATH = PROJECT_DIR / "dsp_interview_transcripts/pipeline/prompts/llm_check_system_a.txt"
OUTPUT_DIR = PROJECT_DIR / "outputs/llm_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def format_row(row):
    """Paste specific cols from a df into a text format that we can feed to GPT"""
    return f" {row['uuid']} | {row['role']} | {row['text']} "


def convert_transcripts_df_to_dict(data):
    """Returns a dict where the keys are conversation IDs and the values are the full text of the conversation."""
    # Convert each interview into one string
    conversation_texts = data.groupby("conversation").apply(lambda group: "\n".join(group.apply(format_row, axis=1)))

    return conversation_texts.to_dict()


def normalize_uuid(val):
    if pd.isnull(val):
        return None
    return str(val).strip()


def main(production: ("Run in production mode", "flag", "p")):  # type: ignore
    data = get_raw_transcripts_cleaned()
    data["uuid"] = data["uuid"].apply(normalize_uuid)

    conversation_text_dict = convert_transcripts_df_to_dict(data)

    system_template = Path(PROMPT_PATH).read_text()

    question_prompt_dict = {}

    question_dict = {
        "desire_more_info": "Did the user seem to desire more information about the trial?",
        "satisfied_temp": "Was the user satisfied with the temperature of their house?",
        "found_house_cold": "Did the user find the house cold as a result of the trial?",
        "accept_automation": "How did the user feel about conceding control of their heating?",
        "noticed_temp_control": "Did the user notice that their temperature was being controlled?",
        "overrode_temp_control": "Did the user override the temperature control?",
        "feedback_ai_interviewer": "Did the user express positive, negative or neutral opinions about the interview process?",
    }

    for k, v in question_dict.items():
        question_prompt_dict[k] = {}
        system_message = system_template.format(question=v)

        if k in ["accept_automation", "feedback_ai_interviewer"]:
            main_field = {
                "name": k,
                "type": "str",
                "description": "Must be either 'Positive', 'Neutral' or 'Negative'.",
            }
        else:
            main_field = {"name": k, "type": "str", "description": "A one-word answer: 'yes' or 'no'."}

        fields = [
            main_field,
            {"name": "explanation", "type": "str", "description": "Explain why you answered in the way you did."},
            {
                "name": "text",
                "type": "List[str]",
                "description": "The text(s) in the transcript where you found the answer.",
            },
            {
                "name": "identifier",
                "type": "List[str]",
                "description": "The identifier(s) of text(s) in the transcript where you found the answer.",
            },
        ]

        question_prompt_dict[k]["system_message"] = system_message
        question_prompt_dict[k]["fields"] = fields

    # Run the process for each question x every conversation in the data
    # (but only one conversation if just testing)
    if not production:
        test_conv = list(conversation_text_dict.keys())[0]
        conversation_text_dict = {test_conv: conversation_text_dict[test_conv]}

    for q in question_prompt_dict:

        if production:
            outpath = f"{OUTPUT_DIR}/{q}_output.jsonl"
        else:
            outpath = f"{OUTPUT_DIR}/{q}_output_test.jsonl"

        processor = batch_check.LLMProcessor(
            model_name="gpt-4o-mini",
            temperature=0,
            output_path=outpath,
            system_message=question_prompt_dict[q]["system_message"],
            session_name=q,
            output_fields=question_prompt_dict[q]["fields"],
        )

        processor.run(conversation_text_dict, batch_size=1, sleep_time=0.5)


if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    plac.call(main)
