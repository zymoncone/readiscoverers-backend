import os
import pandas as pd
import numpy as np
import re
from google.genai.types import EmbedContentConfig
from .constants import EMBEDDING_MODEL_ID


def find_best_passage(query: str, dataframe: pd.DataFrame, client, top_k=3):
    """
    Compute the distances between the query and each document in the dataframe
    using the dot product.
    """
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
        for i, idx in enumerate(top_indices, 1):
            print(f"Result {i} - Dot product: {dot_products[idx]:.4f}")

    results = []
    for idx in top_indices:
        title = str(dataframe.iloc[idx]["title"])
        chunk_text = str(dataframe.iloc[idx]["text"])
        chapter_index = int(dataframe.iloc[idx]["chapter_index"])

        # Extract chapter number and name from title
        # Pattern: "Chapter {number}: {name} ({chunk_num})"
        title_match = re.match(r"Chapter\s+(.+?):\s+(.+?)\s*\((\d+)\)", title)
        if title_match:
            chapter_number_raw = str(title_match.group(1))  # e.g., "I", "II", "1", "2"
            chapter_name = str(title_match.group(2))  # e.g., "The Cyclone"
            chunk_in_chapter_index = int(title_match.group(3))  # e.g., "1", "2"
        else:
            # Handle Introduction or other formats
            chapter_number_raw = None
            chapter_name = title
            chunk_in_chapter_index = None

        # Extract actual text content (everything after "From Chapter ... : ")
        text_match = re.search(r"From Chapter\s+.+?:\s*(.+)", chunk_text, re.DOTALL)
        if text_match:
            actual_text = str(text_match.group(1).strip())
        else:
            # If pattern doesn't match, use full text
            actual_text = str(chunk_text)

        results.append(
            {
                "chunk_index": int(idx),
                "chapter_index": chapter_index,
                "chunk_in_chapter_index": chunk_in_chapter_index,
                "chapter_number_raw": chapter_number_raw,
                "chapter_name": chapter_name,
                "text": actual_text,
                "score": float(dot_products[idx]),
            }
        )

    return results
