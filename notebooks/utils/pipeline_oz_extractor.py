import pandas as pd


def preprocess_oz_extractor_results(
    oz_extractor_results_df: pd.DataFrame,
) -> pd.DataFrame:
    # Extract filenames for all loc columns
    for i in [1, 2, 3]:
        oz_extractor_results_df[f"filename_{i}"] = oz_extractor_results_df[
            f"loc_{i}"
        ].str.extract(r"file:(.+?)\.html")
        oz_extractor_results_df[f"filename_{i}"] = (
            oz_extractor_results_df[f"filename_{i}"]
            .fillna("")
            .apply(
                lambda x: x.lower().replace("-", "_").replace(" ", "_") if x else None
            )
        )

    oz_extractor_results_df.drop(columns=["loc_1", "loc_2", "loc_3"], inplace=True)

    reshaped_rows = []
    for question_num, (_, row) in enumerate(oz_extractor_results_df.iterrows(), 1):
        for rank in [1, 2, 3]:
            reshaped_rows.append(
                {
                    "question": row["question"],
                    "question_number": question_num,
                    "result_rank": rank,
                    "excerpt": row[f"excerpt_{rank}"],
                    "filename": row[f"filename_{rank}"],
                }
            )

    return pd.DataFrame(reshaped_rows)
