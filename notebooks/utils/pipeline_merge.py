"""Merge model output with expected data and check for correctness."""

import ast
import pandas as pd


def merge_model_and_expected_data(
    oz_extractor_results_df: pd.DataFrame,
    expected_chunk_data_df: pd.DataFrame,
    selected_questions: pd.DataFrame = None,
) -> pd.DataFrame:
    """Merge model output with expected data and check for correctness."""

    # Create a mapping from results_df with the columns you need
    results_mapping = expected_chunk_data_df[
        ["original_query", "expected_all_chunk_indices", "expected_book_filename"]
    ].copy()
    results_mapping["original_query"] = results_mapping["original_query"].str.strip()
    results_mapping = results_mapping.drop_duplicates(subset="original_query")

    # Strip whitespace from question column in oz_extractor_results_df
    oz_extractor_results_df["question"] = oz_extractor_results_df[
        "question"
    ].str.strip()

    # Merge with oz_extractor_results_df
    combined_df = oz_extractor_results_df.merge(
        results_mapping, left_on="question", right_on="original_query", how="left"
    )

    # Drop the duplicate original_query column
    combined_df = combined_df.drop(columns=["original_query"])

    combined_df["correct_match"] = combined_df.apply(check_match_and_overlap, axis=1)

    if selected_questions is not None:
        # Get the list of questions from selected_questions if provided
        selected_question_list = selected_questions["Question"].str.strip().tolist()

        # Filter combined_df to only include those questions
        combined_df = combined_df[combined_df["question"].isin(selected_question_list)]

    return combined_df


def check_match_and_overlap(row) -> bool:
    """Check if filenames match and chunks overlap"""
    # Check if both required fields are not null
    if pd.isna(row["model_filename"]) or pd.isna(row["expected_book_filename"]):
        return False

    # Check if filenames match
    if row["model_filename"] != row["expected_book_filename"]:
        return False

    # Handle the case where chunk lists might be strings or actual lists
    model_chunks = row["all_model_chunks"]
    expected_chunks = row["expected_all_chunk_indices"]

    # Convert string representations to actual lists if needed
    if isinstance(model_chunks, str):
        try:
            model_chunks = ast.literal_eval(model_chunks)
        except (ValueError, SyntaxError):
            return False

    if isinstance(expected_chunks, str):
        try:
            expected_chunks = ast.literal_eval(expected_chunks)
        except (ValueError, SyntaxError):
            return False

    # Check if either is None or not a list
    if not isinstance(model_chunks, list) or not isinstance(expected_chunks, list):
        return False

    # Check if lists are empty
    if len(model_chunks) == 0 or len(expected_chunks) == 0:
        return False

    # Check for intersection
    overlap = set(model_chunks) & set(expected_chunks)

    return len(overlap) > 0
