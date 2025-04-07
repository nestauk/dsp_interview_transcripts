from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd

from discovery_utils.utils.llm import batch_check


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
        processor.run(conversation_text_dict, batch_size=1, sleep_time=0.5)
        output_paths[qid] = str(outpath)
    return output_paths
