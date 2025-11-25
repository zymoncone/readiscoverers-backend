"""Module for searching book passages using semantic similarity."""

import os
import re

import pandas as pd
import numpy as np
from google.genai.types import EmbedContentConfig

from .constants import EMBEDDING_MODEL_ID


def find_best_text_chunks(
    query: str,
    dataframe: pd.DataFrame,
    client,
    top_k: int = 3,
    query_id: str = None,
    enhanced_query: bool = None,
    chunking_metadata: dict = None,
) -> list[dict]:
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
    # pylint: disable=too-many-locals
    query_embedding = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=query,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )

    dot_products = np.dot(
        np.stack(dataframe["embeddings"]), query_embedding.embeddings[0].values
    )
    # Get indices of top_k highest dot products
    top_indices = np.argsort(dot_products)[-top_k:][::-1]  # Sort descending

    # Print scores for each result
    if os.environ.get("ENV") == "dev":
        for i, chunk_index in enumerate(top_indices, 1):
            print(f"Result {i} - Dot product: {dot_products[chunk_index]:.4f}")

    # Calculate totals for progress indicators
    total_chunks = len(dataframe)

    results = []
    for similarity_score_order, chunk_index in enumerate(top_indices, 1):
        chapter_title = str(dataframe.iloc[chunk_index]["title"])
        chunk_text = str(dataframe.iloc[chunk_index]["text"])
        chapter_index = int(dataframe.iloc[chunk_index]["chapter_index"])

        chunk_in_chapter_match = re.search(r"\((\d+)\)", chapter_title)
        if chunk_in_chapter_match and chunk_in_chapter_match.group(1).isdigit():
            chunk_in_chapter_index = int(chunk_in_chapter_match.group(1))
        else:
            chunk_in_chapter_index = None

        # Extract actual text content (everything after "From Chapter ... : ")
        text_match = re.search(r"From Chapter\s+.+?:\s*(.+)", chunk_text, re.DOTALL)
        if text_match:
            actual_text = str(text_match.group(1).strip())
        else:
            # If pattern doesn't match, use full text
            actual_text = str(chunk_text)

        # Calculate progress indicators
        book_progress = round((int(chunk_index) / total_chunks) * 100, 1)

        result = {
            # For logging/tracking
            "query_id": query_id,
            "query": query,
            "enhanced_query": enhanced_query,
            # Results data
            "chunk_index": int(chunk_index),
            "chapter_index": chapter_index,
            "chunk_in_chapter_index": chunk_in_chapter_index,
            "chapter_number": chapter_index,
            "chapter_title": chapter_title,
            "text": actual_text,
            "score": float(dot_products[chunk_index]),
            "book_progress_percent": book_progress,
            "passage_number": f"Passage {int(chunk_index) + 1} of {total_chunks}",
        }

        # Add chunking metadata if provided
        if chunking_metadata:
            result.update(chunking_metadata)

        results.append({"score_order": int(similarity_score_order), "data": result})

    return results
