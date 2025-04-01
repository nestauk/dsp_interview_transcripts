import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


def get_preceding_context(topics_df, interview_df, context_window=3):
    results = []
    for _, row in topics_df.iterrows():
        # Filter for matching file_name in the interview dataframe
        filtered_df = interview_df[interview_df["file_name"] == row["file_name"]]

        # Find the index of the matching text
        match_idx = filtered_df[filtered_df["text"] == row["text"]].index

        if not match_idx.empty:
            idx = match_idx[0]
            # Extract the preceding context
            start_idx = max(0, idx - context_window)
            context = filtered_df.loc[start_idx:idx]
            context_with_roles = [{row["role"]: row["text"]} for _, row in context.iterrows()]
            context_formatted = " ".join(
                [f"{role}: {text}" for context in context_with_roles for role, text in context.items()]
            )
            results.append(
                {
                    "text": row["text"],
                    "file_name": row["file_name"],
                    "preceding_context": context_with_roles,
                    "context_formatted": context_formatted,
                }
            )
        else:
            results.append(
                {
                    "text": row["text"],
                    "file_name": row["file_name"],
                    "preceding_context": None,
                    "context_formatted": None,
                }
            )
    return pd.merge(topics_df, pd.DataFrame(results), on=["text", "file_name"], how="left")


if __name__ == "__main__":
    maquettes_df = pd.read_csv(PROJECT_DIR / "data/bit_france/converted/maquettes_df.csv")
    maquettes_df["role"] = maquettes_df["role"].replace("other", "INTERVIEWER")
    maquettes_df["role"] = maquettes_df["role"].replace("informant", "INFORMANT")

    result_df = get_preceding_context(maquettes_df, maquettes_df)

    result_df.to_csv(PROJECT_DIR / "data/bit_france/converted/maquettes_df_context.csv", index=False)
