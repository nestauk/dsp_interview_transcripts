import asyncio
import json
import time
import uuid

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd

from discovery_utils.utils.llm import batch_check

from dsp_interview_transcripts import PROJECT_DIR


PROMPT_PATH = PROJECT_DIR / "dsp_interview_transcripts/pipeline/prompts/llm_check_system_a.txt"


def parse_rqs(rq_text: str) -> Dict[str, str]:
    """Given research questions entered as one string,
    split these line by line and return a dict
    that maps an arbitrary ID for each RQ to the text of the question.

    Args:
        rq_text (str): One string containing all the RQs.

    Returns:
        Dict[str, str]: Dict mapping an ID to the text of the RQ.
    """
    research_questions = rq_text.strip().splitlines()
    rq_dict = {f"rq_{i+1}": q for i, q in enumerate(research_questions)}
    return rq_dict


def format_row(row: pd.Series, role_col: str, text_col: str, uuid_col: str) -> str:
    """Convert each row from a dataframe into a formatted string.
    This is because we can't submit a dataframe in a prompt to the LLM, so we have
    to convert the dataframe to a string that follows a predictable structure.

    Args:
        row (pd.Series): A row of the DataFrame.
        role_col (str): Column name for the speaker role (BOT or USER).
        text_col (str): Column name for the text content.
        uuid_col (str): Column name for the unique text identifier.

    Returns:
        str: A formatted string in the form "uuid | role | text".
    """
    return f"{row[uuid_col]} | {row[role_col]} | {row[text_col]}"


def convert_transcripts_df_to_dict(
    data: pd.DataFrame, conv_id_col: str, role_col: str, text_col: str, uuid_col: str
) -> Dict[Any, str]:
    """
    Convert a DataFrame where each row is a line from a conversation,
    into a dict where the keys are conversation IDs and the values are
    formatted strings containing the entire conversation line by line.

    Args:
        data (pd.DataFrame): The transcript DataFrame.
        conv_id_col (str): Column name for conversation IDs.
        role_col (str): Column name for the speaker role.
        text_col (str): Column name for the text content.
        uuid_col (str): Column name for the unique text identifier.

    Returns:
        Dict[Any, str]: A dictionary where keys are conversation IDs and values are the transcripts of each conversation.
    """
    return (
        data.groupby(conv_id_col)
        .apply(lambda group: "\n".join(group.apply(lambda row: format_row(row, role_col, text_col, uuid_col), axis=1)))
        .to_dict()
    )


def normalize_uuid(val: Any) -> Optional[str]:
    """
    Normalize a UUID value by converting it to a stripped string, or return None if null.
    Not used right now but in future it would be nice to validate the LLM output
    and check it has not hallucinated any IDs - this function will help compare IDs.

    Args:
        val (Any): A UUID or null value.

    Returns:
        Optional[str]: A normalized string UUID, or None.
    """
    if pd.isnull(val):
        return None
    return str(val).strip()


def build_question_prompt_dict(
    rq_dict: Dict[str, str], prompt_template_text: str
) -> Dict[str, Dict[str, Union[str, list]]]:
    """
    Constructs the output fields for each RQ.

    At the moment the only difference this makes is to have one field that is named the same thing
    as the RQ, but in future this can be expanded so that different RQs generate different outputs
    (e.g. sentiment-related ones generating "Positive"/"Negative"/"Neutral" rather than "yes"/"no").

    Args:
        rq_dict (Dict[str, str]): Mapping of question IDs to question text.
        prompt_template_text (str): The system prompt template, with {question} as a placeholder.

    Returns:
        Dict[str, Dict[str, Union[str, list]]]: A dictionary where each key is a question ID,
            and each value contains a system message and output field definitions.
    """
    prompt_dict = {}
    for k, question in rq_dict.items():
        system_message = prompt_template_text.format(question=question)
        fields = [
            {"name": k, "type": "str", "description": "A one-word answer: 'yes' or 'no'."},
            {"name": "explanation", "type": "str", "description": "Explain why you answered in the way you did."},
            {"name": "text", "type": "List[str]", "description": "Relevant text from transcript."},
            {"name": "identifier", "type": "List[str]", "description": "Text identifiers."},
        ]
        prompt_dict[k] = {"system_message": system_message, "fields": fields}
    return prompt_dict


def get_mock_outputs_for_each_rq(
    df: pd.DataFrame, rq_dict: Dict[str, str], uuid_col: str, output_dir: str
) -> Dict[str, str]:
    """Mocks the LLM batch_check outputs.

    Args:
        df (pd.DataFrame): The input DataFrame containing the "text_clean" and UUID columns.
        rq_dict (Dict[str, str]): A mapping from RQ ID to question text.
        uuid_col (str): The name of the column containing unique IDs for each row.
        output_dir (str): Directory path where the mock JSONL files will be saved.

    Returns:
        Dict[str, str]: A mapping from RQ ID to the path of the corresponding (mock or existing) JSONL file.
    """
    output_paths = {}
    for rq_id in rq_dict:
        mock_path = Path(f"{output_dir}/{rq_id}_output.jsonl")
        if mock_path.exists():
            output_paths[rq_id] = str(mock_path)
        else:
            # grab some random quotes from the original data
            sample_df = df.sample(n=3)
            mock_quotes = sample_df["text_clean"].tolist()
            mock_ids = sample_df[uuid_col].tolist()

            mock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mock_path, "w") as f:
                for quote, q_id in zip(mock_quotes, mock_ids):
                    f.write(
                        json.dumps(
                            {
                                "question": rq_id,
                                "answer": "This is a test explanation",
                                "text": [quote],
                                "identifier": [q_id],
                                "id": str(uuid.uuid4()),
                                "timestamp": "2025-04-07T00:00:00Z",
                                "model": "mock",
                                "temperature": 0,
                            }
                        )
                        + "\n"
                    )
            output_paths[rq_id] = str(mock_path)
    return output_paths


def run_batch_check(
    conversation_text_dict: Dict[Any, str],
    prompt_dict: Dict[str, Dict[str, Union[str, list]]],
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """
    Run the batch_check pipeline for each research question on the conversation data.

    Args:
        conversation_text_dict (Dict[Any, str]): Mapping of conversation IDs to full transcript strings.
        prompt_dict (Dict[str, Dict[str, Union[str, list]]]): Dict containing the system message and desired output fields for each RQ.
        output_dir (Union[str, Path]): Directory to write output `.jsonl` files (one per RQ).

    Returns:
        Dict[str, str]: A mapping from research question IDs to their respective output file paths.
    """

    output_paths = {}
    for qid, meta in prompt_dict.items():
        outpath = Path(output_dir) / f"{qid}_output.jsonl"
        processor = batch_check.LLMProcessor(
            model_name="gpt-4o-mini",
            temperature=0,
            output_path=str(outpath),
            system_message=meta["system_message"],
            session_name=qid,
            output_fields=meta["fields"],
        )
        processor.run(conversation_text_dict, batch_size=50, sleep_time=0.5)
        output_paths[qid] = str(outpath)
    return output_paths


def concat_batch_check_output(rq_dict: Dict[str, str], output_paths: Dict[str, str]) -> pd.DataFrame:
    """Concatenate the outputs from the batch_check process,
    giving one dataframe with one row per conversation * per RQ.

    Args:
        rq_dict (Dict[str, str]): A mapping from RQ ID to question text.
        output_paths (Dict[str, str]): A mapping from RQ ID to the path of its corresponding JSONL output file.

    Returns:
        pd.DataFrame: A DataFrame containing the concatenated outputs from all RQs.
                Each row corresponds to a conversation and contains the RQ answer and other relevant fields.
                Total length = one row per RQ * per conversation.
    """
    step1_output_df = pd.DataFrame()

    for rq_id, question in rq_dict.items():
        path = output_paths[rq_id]

        temp_df = pd.read_json(path, lines=True)

        # Find the column that starts with "rq_"
        rq_column = next(col for col in temp_df.columns if col.startswith("rq_"))

        # Rename that column to "answer"
        temp_df = temp_df.rename(
            columns={
                rq_column: "answer",
                # "id": "conversation_id"
            }
        )

        temp_df["rq"] = question

        step1_output_df = pd.concat([step1_output_df, temp_df], ignore_index=True)

    return step1_output_df


def run_batch_check_for_all_rqs(
    rq_text: str,
    cleaned_df: pd.DataFrame,
    output_dir: str,
    conv_col: str,
    role_col: str,
    uuid_col: str,
    prompt_path: Path = PROMPT_PATH,
    test_mode: bool = False,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Brings together the functions above to run batch_check for each RQ.

    Args:
        rq_text (str): Raw research question input string.
        cleaned_df (pd.DataFrame): DataFrame with conversation data.
        output_dir (str): Directory where results (real or mock) are saved.
        conv_col (str): Column name for conversation ID.
        role_col (str): Column name for speaker role (e.g., 'user' or 'bot').
        uuid_col (str): Column with unique identifiers for each message.
        prompt_path (Path): Path to the prompt template file.
        test_mode (bool): If True, generate mock outputs instead of running the real batch check.

    Returns:
        Tuple[Dict[str, str], pd.DataFrame]:
            - Dictionary mapping RQ IDs to their output file paths.
            - DataFrame with batch_check results for each conversation and for each RQ.
    """

    rq_dict = parse_rqs(rq_text)

    prompt_template = prompt_path.read_text()
    conversation_dict = convert_transcripts_df_to_dict(cleaned_df, conv_col, role_col, "text_clean", uuid_col)
    prompt_dict = build_question_prompt_dict(rq_dict, prompt_template)

    if test_mode:
        output_paths = get_mock_outputs_for_each_rq(cleaned_df, rq_dict, uuid_col, output_dir)
    else:
        output_paths = run_batch_check(conversation_dict, prompt_dict, output_dir)

    # NOTE: I would like concat_batch_check_output to be wrapped up in this function but I ran into
    # an issue where concat_batch_check_output would try to run before the asyncio loop had
    # finished writing the files. So I have commented it out for now.
    # df_output = concat_batch_check_output(rq_dict, output_paths)

    return output_paths, rq_dict
