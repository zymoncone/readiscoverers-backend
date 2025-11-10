import os
from typing import Optional, Union
import re
import pandas as pd
import requests
from google.genai.types import EmbedContentConfig

from .constants import EMBEDDING_MODEL_ID, TEMP_DIR


def download_file(url: str, local_filename: str) -> dict:
    """Download a file from URL to local workspace"""
    if url is None or not url.endswith(".txt"):
        return {
            "status": "error",
            "message": "A valid .txt URL must be provided.",
        }
    if not local_filename.endswith(".txt"):
        filename_ext = f"{local_filename}.txt"
        if os.environ.get("ENV") == "dev":
            print("Appended .txt to local filename:", filename_ext)
    else:
        filename_ext = local_filename
        if os.environ.get("ENV") == "dev":
            print("Using provided local filename:", filename_ext)
    try:
        # Check if 'temp' exists as a file and remove it
        if os.path.exists(TEMP_DIR) and os.path.isfile(TEMP_DIR):
            os.remove(TEMP_DIR)

        # Create temp directory
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Construct full path with temp/ directory
        filepath = os.path.join(TEMP_DIR, filename_ext)

        response = requests.get(url)
        response.raise_for_status()  # Raise error if download fails

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)

        if os.environ.get("ENV") == "dev":
            print(f"Downloaded {filepath}")
        return {"status": "success", "file": filepath}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error downloading {filename_ext}: {str(e)}",
        }


def embed_fn(title: str, text: str, client) -> Union[dict, None]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", title=title),
    )

    return response.embeddings[0].values


def apply_embeddings(df: pd.DataFrame, client) -> Union[pd.DataFrame, None]:
    if df.empty:
        if os.environ.get("ENV") == "dev":
            print("DataFrame is empty; skipping embedding application.")
        return None
    df["embeddings"] = df.apply(
        lambda row: embed_fn(row["title"], row["text"], client), axis=1
    )
    return df


def read_book_to_chunks(local_filename: str, chunk_size: int) -> pd.DataFrame:
    filepath = os.path.join(TEMP_DIR, f"{local_filename}.txt")
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        book = f.read()

    # Find the start of the actual content
    intro_matches = list(re.finditer(r"Introduction", book))

    if len(intro_matches) >= 2:
        start_pos = intro_matches[1].start()
    else:
        start_pos = (
            intro_matches[0].start()
            if intro_matches
            else re.search(r"\*\*\*\s*START OF(.*?)\*\*\*", book, re.IGNORECASE).start()
        )

    print("Start pos:", start_pos)

    end_match = re.search(r"\*\*\*\s*END OF(.*?)\*\*\*", book, re.IGNORECASE)
    end_pos = end_match.start() if end_match else len(book)

    print("End pos:", end_pos)

    content = book[start_pos:end_pos]

    # Pattern to match chapter headings with number and name
    # Try two patterns:
    # 1. Title below chapter number: "CHAPTER 1\nThe Title"
    # 2. Title above chapter number: "The Title\n\nCHAPTER 1"

    # Pattern 1: Title below (like "CHAPTER 1\nThe Title")
    chapter_pattern_below = r"^Chapter\s+((?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty))\s*\n(.+?)$"

    # Pattern 2: Title above (like "The Title\n\nCHAPTER 1")
    chapter_pattern_above = r"^([A-Z][A-Za-z\s]{2,50})\s*\n+\s*Chapter\s+((?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty))\s*$"

    # Try pattern 1 first (title below - most common)
    chapter_matches = list(
        re.finditer(chapter_pattern_below, content, flags=re.MULTILINE | re.IGNORECASE)
    )
    title_position = "below"  # Track which pattern matched

    # If no matches, try pattern 2 (title above)
    if not chapter_matches:
        chapter_matches = list(
            re.finditer(
                chapter_pattern_above, content, flags=re.MULTILINE | re.IGNORECASE
            )
        )
        title_position = "above"

    if not chapter_matches:
        print("NO CHAPTERS FOUND. Check the chapter pattern or the book format.")
        return pd.DataFrame(columns=["chapter_index", "title", "text"])

    documents = []

    # Extract text between chapters
    for chapter_index, match in enumerate(chapter_matches):
        chapter_start = match.start()

        # Extract chapter number and name based on which pattern matched
        if title_position == "above":
            chapter_name = match.group(1).strip()
            chapter_number_raw = match.group(2).strip()
        else:  # below
            chapter_number_raw = match.group(1).strip()
            chapter_name = match.group(2).strip()

        new_chunk_starter_text = f"From Chapter {chapter_number_raw} {chapter_name}: "
        full_title = f"Chapter {chapter_number_raw}: {chapter_name}"

        if os.environ.get("ENV") == "dev":
            print(f"Found chapter: {full_title}")

        # Get the text until the next chapter (or end of content)
        if chapter_index + 1 < len(chapter_matches):
            chapter_end = chapter_matches[chapter_index + 1].start()
        else:
            chapter_end = len(content)

        chapter_text = content[chapter_start:chapter_end].strip()

        paragraph_chunks = [p.strip() for p in chapter_text.split("\n\n") if p.strip()]

        new_chunk = new_chunk_starter_text
        chunk_index = 0
        for paragraph in paragraph_chunks:
            new_chunk += paragraph + "\n\n"
            if len(new_chunk) >= chunk_size:
                documents.append(
                    (
                        int(chapter_index),
                        f"{full_title} ({chunk_index})",
                        new_chunk.strip(),
                    )
                )
                new_chunk = new_chunk_starter_text
                chunk_index += 1
        if new_chunk:
            documents.append(
                (
                    int(chapter_index),
                    f"{full_title} ({chunk_index})",
                    new_chunk.strip(),
                )
            )

    # Add introduction if it exists before first chapter
    if chapter_matches and intro_matches:
        intro_text = content[: chapter_matches[0].start()].strip()
        # TODO: do the same chunking for intro as for chapters
        if len(intro_text) > 100:
            documents.insert(0, (0, "Introduction", intro_text))

    if os.environ.get("ENV") == "dev":
        print(f"Number of chunks: {len(documents)}")

    df = pd.DataFrame(documents)
    df.columns = ["chapter_index", "title", "text"]

    return df


def get_book_df(
    url: str = None, local_filename: str = None, chunk_size: int = 1200, client=None
) -> Union[pd.DataFrame, dict]:
    if url is None or local_filename is None:
        return {
            "status": "error",
            "message": "URL and local filename must be provided.",
        }
    response = download_file(url, local_filename)
    if response["status"] == "error":
        return response
    if client is None:
        return {
            "status": "error",
            "message": "GenAI client must be provided.",
        }
    data = apply_embeddings(read_book_to_chunks(local_filename, chunk_size), client)
    if data is None:
        return {
            "status": "error",
            "message": "No data to process after reading book. Check parsing logic.",
        }
    return {
        "status": "success",
        "data": data,
    }
