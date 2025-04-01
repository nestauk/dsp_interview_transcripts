import pandas as pd

from dsp_interview_transcripts import PROJECT_DIR
from dsp_interview_transcripts import logger


def filter_after_new_section(group):
    # Find the index of the first occurrence of `new_section == 1`
    new_section_idx = group[group["new_section"] == "1"].index
    if not new_section_idx.empty:
        # Keep rows after the first occurrence
        return group.loc[new_section_idx[0] + 1 :]
    return pd.DataFrame()  # Return an empty DataFrame if no `new_section == 1` is found


if __name__ == "__main__":
    labelled_data = pd.read_csv(
        PROJECT_DIR
        / "data/bit_france/converted/bit_france_interview_sections - ALT_data_labelled_with_new_sections.csv"
    )

    filtered_data = labelled_data[
        ["speaker_id", "text", "file_name", "profession", "informant", "role", "new_section"]
    ].query('profession != "Médecins du travail"')

    maquettes_df = filtered_data.groupby("file_name", group_keys=False).apply(filter_after_new_section)

    logger.info(
        f"Difference in unique files after filtering (should be 1): {len(filtered_data['file_name'].unique()) - len(maquettes_df['file_name'].unique())}"
    )

    maquettes_df.to_csv(PROJECT_DIR / "data/bit_france/converted/maquettes_df.csv", index=False)
