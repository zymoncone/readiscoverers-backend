"""Module for searching book passages using semantic similarity."""

import os
import re

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
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
    keywords: list = None,
    context_chunks: int = 1,
    fuzzy_threshold: int = 80,
) -> dict:
    """
    Find the best matching text chunks for a given query using semantic similarity.

    Take advantage of keywords or enhanced query context to improve matching accuracy
    by narrowing down to a specific book when possible.

    Args:
        query (str): The search query.
        combined_books_df (pd.DataFrame): DataFrame containing book chunks with embeddings.
        client: The embedding client to use.
        top_k (int): Number of top results to return.
        query_id (str): Optional identifier for the query.
        enhanced_query (bool): Whether the query has been enhanced with additional context.
        chunking_metadata (dict): Optional metadata to include in results.
        keywords (list): Optional list of keywords extracted from the query.
        context_chunks (int): Number of surrounding chunks to include for context.
        fuzzy_threshold (int): Minimum fuzzy match score to consider a book title match.
    """

    # Check if query mentions a specific book title (fuzzy matching)
    unique_books = combined_books_df["book_title"].unique()
    matched_book = None
    highest_score = 0

    query_lower = query.lower()

    if enhanced_query and keywords:
        # Use keywords but filter out very short ones that cause false matches
        # Only use keywords that are likely full book titles (longer phrases)
        filtered_keywords = [kw for kw in keywords if len(kw.split()) >= 3]
        texts_to_check = (
            [query_lower] + [kw.lower() for kw in filtered_keywords]
            if filtered_keywords
            else [query_lower]
        )

    else:
        # Extract potential book titles from the query using regex
        title_patterns = [
            r"In\s+([^,]+?),",  # "In The Emerald City of Oz,"
            r"from\s+([^,]+?)[,\?]",  # "from The Emerald City of Oz,"
            r"in\s+the\s+book\s+([^,]+?)[,\?]",  # "in the book The Emerald City of Oz"
        ]

        extracted_titles = []
        for pattern in title_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match and len(match.group(1).strip()) >= 3:
                extracted_titles.append(match.group(1).strip())

        # Only use extracted titles if we found any, otherwise use the full query
        texts_to_check = extracted_titles if extracted_titles else [query_lower]

    print("Texts to check for fuzzy matching:", texts_to_check)

    for text in texts_to_check:
        for book_title in unique_books:
            book_title_lower = str(book_title).lower()
            # Use token_set_ratio which is better for handling "the", "a", etc.
            # and doesn't give false positives for short substring matches
            score = fuzz.token_set_ratio(text, book_title_lower)

            if score > highest_score and score >= fuzzy_threshold:
                highest_score = score
                matched_book = book_title

    # Filter dataframe if a book was matched
    search_df = combined_books_df
    if matched_book:
        search_df = combined_books_df[combined_books_df["book_title"] == matched_book]
        if os.environ.get("ENV") == "dev":
            print(
                f"Narrowing search to book: {matched_book} (fuzzy score: {highest_score})"
            )
            print(
                f"Searching {len(search_df)} chunks instead of {len(combined_books_df)}"
            )
    else:
        if os.environ.get("ENV") == "dev":
            print(
                "No specific book matched; searching all books. Best score was:",
                highest_score,
            )

    # pylint: disable=too-many-locals
    query_embedding = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=query,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )

    dot_products = np.dot(
        np.stack(search_df["embeddings"]), query_embedding.embeddings[0].values
    )
    # Get indices of top_k highest dot products
    top_indices = np.argsort(dot_products)[-top_k:][::-1]  # Sort descending

    results = []
    for similarity_score_order, chunk_index in enumerate(top_indices, 1):
        row = search_df.iloc[chunk_index]

        chapter_title = str(row["title"])
        chapter_index = int(row["chapter_index"])
        filename = row["filename"]
        book_chunk_index = int(row["book_chunk_index"])
        book_chunk_length = int(row["book_chunk_length"])
        book_title = str(row["book_title"])
        book_author = str(row["book_author"])

        # Get surrounding chunks for context
        book_df = search_df[search_df["filename"] == filename]

        # Get context window (previous and next chunks)
        start_idx = max(0, book_chunk_index - context_chunks)
        end_idx = min(len(book_df), book_chunk_index + context_chunks + 1)

        context_rows = book_df.iloc[start_idx:end_idx]

        # Build context with highlight markers
        matched_texts = []
        for _, ctx_row in context_rows.iterrows():
            ctx_text = str(ctx_row["text"])
            # Extract actual text content
            text_match = re.search(
                r"Book:\s+[^,]+,\s+Chapter:\s+\d+\s+[^-]+-\s*(.+)", ctx_text, re.DOTALL
            )
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
