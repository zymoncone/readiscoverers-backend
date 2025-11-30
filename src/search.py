"""Module for searching book passages using semantic similarity."""

import os
import re

import pandas as pd
import numpy as np
from google.genai.types import EmbedContentConfig

from .constants import EMBEDDING_MODEL_ID


def find_best_text_chunks(
    query: str,
    combined_books_df: pd.DataFrame,
    client,
    top_k: int = 3,
    query_id: str = None,
    enhanced_query: bool = None,
    chunking_metadata: dict = None,
    context_chunks: int = 1,
) -> dict:
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
        np.stack(combined_books_df["embeddings"]), query_embedding.embeddings[0].values
    )
    # Get indices of top_k highest dot products
    top_indices = np.argsort(dot_products)[-top_k:][::-1]  # Sort descending

    results = []
    for similarity_score_order, chunk_index in enumerate(top_indices, 1):
        row = combined_books_df.iloc[chunk_index]

        chapter_title = str(row["title"])
        chapter_index = int(row["chapter_index"])
        filename = row["filename"]
        book_chunk_index = int(row["book_chunk_index"])
        book_chunk_length = int(row["book_chunk_length"])
        book_title = str(row["book_title"])
        book_author = str(row["book_author"])

        # Get surrounding chunks for context
        book_df = combined_books_df[combined_books_df["filename"] == filename]

        # Get context window (previous and next chunks)
        start_idx = max(0, book_chunk_index - context_chunks)
        end_idx = min(len(book_df), book_chunk_index + context_chunks + 1)

        context_rows = book_df.iloc[start_idx:end_idx]

        # Build context with highlight markers
        matched_texts = []
        for _, ctx_row in context_rows.iterrows():
            ctx_text = str(ctx_row["text"])
            # Extract actual text content
            text_match = re.search(r"From Chapter\s+.+?:\s*(.+)", ctx_text, re.DOTALL)
            if text_match:
                ctx_text = text_match.group(1).strip()

            # Remove double-double quotes (pandas CSV escaping artifact)
            ctx_text = ctx_text.replace('""', '"')

            if ctx_row["book_chunk_index"] == book_chunk_index:
                # This is the highlighted chunk
                matched_texts.append(
                    {
                        "chunk_index": int(ctx_row["book_chunk_index"]),
                        "text": ctx_text,
                        "is_match": True,
                    }
                )
            else:
                matched_texts.append(
                    {
                        "chunk_index": int(ctx_row["book_chunk_index"]),
                        "text": ctx_text,
                        "is_match": False,
                    }
                )

        # Calculate progress based on book-specific values
        book_progress = round((book_chunk_index + 1) / book_chunk_length * 100, 1)

        chunk_in_chapter_match = re.search(r"(.*)\((\d+)\)", chapter_title)
        if chunk_in_chapter_match and chunk_in_chapter_match.group(2).isdigit():
            chunk_in_chapter_index = int(chunk_in_chapter_match.group(2))
            chapter_title = chunk_in_chapter_match.group(1).strip()
        else:
            chunk_in_chapter_index = None

        # Calculate progress indicators
        book_progress = round(
            (book_chunk_index + 1) / book_chunk_length * 100, 1
        )  # total chunks should be book length

        result = {
            # For logging/tracking
            "query_id": query_id,
            "query": query,
            "enhanced_query": enhanced_query,
            # Results data
            "chunk_index": book_chunk_index,
            "chapter_index": chapter_index,  # can get rid of this?
            "chunk_in_chapter_index": chunk_in_chapter_index,
            "chapter_number": chapter_index,
            "chapter_title": chapter_title,
            "matched_texts": matched_texts,
            "score": float(dot_products[chunk_index]),
            "book_progress_percent": book_progress,
            "passage_number": f"Passage {book_chunk_index + 1} of {book_chunk_length}",
            "book_title": book_title,
            "book_author": book_author,
            "filename": filename,
        }

        # Add chunking metadata if provided
        if chunking_metadata:
            result.update(chunking_metadata)

        results.append({"score_order": int(similarity_score_order), "data": result})

    return {"status": "success", "search_results": results}
