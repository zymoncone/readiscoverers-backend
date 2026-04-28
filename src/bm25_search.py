"""Module for BM25 text search functionality."""

import re
import pandas as pd
from rank_bm25 import BM25Okapi


def calculate_bm25_scores(search_df: pd.DataFrame, query: str) -> list[float]:
    """
    Calculate BM25 scores for all chunks in the search dataframe.

    Args:
        search_df (pd.DataFrame): DataFrame containing text chunks to search.
        query (str): The search query.

    Returns:
        list[float]: BM25 scores for each chunk in the dataframe.
    """
    # Extract text content and tokenize for BM25
    corpus = []
    for text in search_df["text"]:
        # Extract actual text content
        text_match = re.search(
            r"Book:\s+[^,]+,\s+Chapter:\s+\d+\s+[^-]+-\s*(.+)", str(text), re.DOTALL
        )
        if text_match:
            clean_text = text_match.group(1).strip()
        else:
            clean_text = str(text)
        # Remove double-double quotes
        clean_text = clean_text.replace('""', '"')
        # Simple tokenization (split on whitespace and lowercase)
        corpus.append(clean_text.lower().split())

    # Create BM25 index
    bm25 = BM25Okapi(corpus)

    # Tokenize query
    tokenized_query = query.lower().split()

    # Get BM25 scores for all documents
    bm25_scores = bm25.get_scores(tokenized_query)

    return bm25_scores
