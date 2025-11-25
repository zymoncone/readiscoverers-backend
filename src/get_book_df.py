"""Module for downloading and processing books into searchable chunks."""

import os
import re
from typing import Union

import pandas as pd
import requests
from google.genai.types import EmbedContentConfig

from .constants import EMBEDDING_MODEL_ID, TEMP_DIR, COLUMN_NAMES
from .parse_html import parse_html_book
from .parse_txt import parse_txt_book


def embed_fn(title: str, text: str, client) -> Union[dict, None]:
    """Generate embeddings for a text chunk using the specified model.

    Args:
        title: The title of the chunk
        text: The text content to embed
        client: The GenAI client instance

    Returns:
        Embedding values or None on error
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL_ID,
        contents=text,
        config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", title=title),
    )

    return response.embeddings[0].values if response.embeddings else None


def apply_embeddings(df: pd.DataFrame, client) -> Union[pd.DataFrame, None]:
    """Apply embeddings to each row in the dataframe.

    Args:
        df: DataFrame containing book chunks
        client: The GenAI client instance

    Returns:
        DataFrame with embeddings column added, or None if empty
    """
    if df.empty:
        if os.environ.get("ENV") == "dev":
            print("DataFrame is empty; skipping embedding application.")
        return None
    df["embeddings"] = df.apply(
        lambda row: embed_fn(row["title"], row["text"], client), axis=1
    )
    return df


def download_file(url: str, local_filename: str) -> dict:
    """Download a file from URL to local workspace.

    Args:
        url: The URL of the text file to download
        local_filename: The local filename to save the downloaded file as
    Returns:
        A dict with 'status' and either 'file' path or error 'message'
    """
    if url is None or not url.endswith(".txt") and not url.endswith(".html"):
        return {
            "status": "error",
            "message": "A valid .txt or .html URL must be provided.",
        }

    file_extension = ".html" if url.endswith(".html") else ".txt"

    if os.environ.get("ENV") == "dev":
        print(f"Determined file extension: {file_extension}")

    # Remove any file extension from local_filename
    local_filename_cleaned = re.sub(r"\.[^./\\]+$", "", local_filename)
    filename_ext = f"{local_filename_cleaned}{file_extension}"

    try:
        # Check if 'temp' exists as a file and remove it
        if os.path.exists(TEMP_DIR) and os.path.isfile(TEMP_DIR):
            os.remove(TEMP_DIR)

        # Create temp directory
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Construct full path with temp/ directory
        filepath = os.path.join(TEMP_DIR, filename_ext)

        response = requests.get(url, timeout=60)  # 60 second timeout
        response.raise_for_status()  # Raise error if download fails

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(response.text)

        if os.environ.get("ENV") == "dev":
            print(f"Downloaded {filepath}")
        return {"status": "success", "filepath": filepath}
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Error downloading {filename_ext}: {str(e)}",
        }
    except IOError as e:
        return {
            "status": "error",
            "message": f"Error writing file {filename_ext}: {str(e)}",
        }


class ChunkProcessor:
    """Class to process book chapters into chunks with various chunking strategies."""

    def __init__(
        self,
        target_chunk_size: int,
        sentence_overlap: int,
        small_paragraph_length: int,
        small_paragraph_overlap: int,
    ):
        self.target_chunk_size = target_chunk_size
        self.sentence_overlap = sentence_overlap
        self.small_paragraph_length = small_paragraph_length
        self.small_paragraph_overlap = small_paragraph_overlap
        self.processed_chunks = []

    def chunk_chapter(
        self, chapter_index: int, chapter_title: str, content: str
    ) -> None:
        """Process a single chapter's content into chunks.

        Args:
            chapter_index: The index of the chapter
            chapter_title: The full title of the chapter
            content: The paragraph content joined with \\n\\n
        """
        # Split content back into paragraphs
        if os.environ.get("ENV") == "dev":
            print(f"Chunking Chapter {chapter_index}: '{chapter_title}'")
        paragraph_chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

        new_chunk_starter_text = f"From Chapter {chapter_index} {chapter_title}: "
        chunk = new_chunk_starter_text
        chunk_index = 0
        chunking_style = None

        for paragraph in paragraph_chunks:
            # If current paragraph is large AND we have accumulated content, save current chunk first
            if len(paragraph) >= self.target_chunk_size and len(chunk) > len(
                new_chunk_starter_text
            ):
                self._write_chunk(
                    chapter_index, chapter_title, chunk_index, chunk, chunking_style
                )
                chunk_index += 1
                chunk = new_chunk_starter_text
                chunking_style = None

            # Handle large paragraphs that need sentence-level splitting
            if len(paragraph) >= self.target_chunk_size:
                chunk, chunk_index = self._process_large_paragraph(
                    chapter_index,
                    chapter_title,
                    chunk_index,
                    chunk,
                    paragraph,
                    new_chunk_starter_text,
                )
                chunking_style = None
            else:
                # Small paragraph - add to accumulating chunk
                if chunking_style != "multi_paragraph_chunk_with_overlap":
                    chunking_style = "multi_paragraph_chunk_no_overlap"
                chunk += paragraph + "\n\n"

                # Check if accumulated chunk has reached target size
                if len(chunk.replace("\n\n", " ").strip()) >= self.target_chunk_size:
                    chunk, chunk_index, chunking_style = (
                        self._write_multiparagraph_chunk(
                            chapter_index,
                            chapter_title,
                            chunk_index,
                            chunk,
                            chunking_style,
                            new_chunk_starter_text,
                        )
                    )

        # Save any remaining chunk content at end of chapter
        if len(chunk) > len(new_chunk_starter_text):
            self._write_chunk(
                chapter_index, chapter_title, chunk_index, chunk, chunking_style
            )

    def _write_chunk(
        self,
        chapter_index: int,
        chapter_title: str,
        chunk_index: int,
        chunk: str,
        chunking_style: str,
    ) -> None:
        """Write a single chunk to processed_chunks."""
        self.processed_chunks.append(
            (
                int(chapter_index),
                f"{chapter_title} ({chunk_index})",
                len(chunk.replace("\n\n", " ").strip()),
                chunking_style,
                chunk.strip(),
            )
        )

    def _write_multiparagraph_chunk(
        self,
        chapter_index: int,
        chapter_title: str,
        chunk_index: int,
        chunk: str,
        chunking_style: str,
        new_chunk_starter_text: str,
    ) -> tuple:
        """Write multi-paragraph chunk with overlap and return new chunk state."""
        self._write_chunk(
            chapter_index, chapter_title, chunk_index, chunk, chunking_style
        )

        # Get paragraphs from the chunk we just saved for overlap
        saved_chunk_paragraphs = [p for p in chunk.split("\n\n") if p.strip()]
        new_chunk = new_chunk_starter_text

        # Add small paragraphs from end of previous chunk as overlap
        paragraph_index = 1
        small_paragraph_chunks = []
        while (
            paragraph_index <= len(saved_chunk_paragraphs)
            and len(saved_chunk_paragraphs[-paragraph_index].strip())
            <= self.small_paragraph_length
            and paragraph_index <= self.small_paragraph_overlap
            and saved_chunk_paragraphs[-paragraph_index].strip()
            != new_chunk_starter_text.strip()
        ):
            small_paragraph_chunks.append(
                saved_chunk_paragraphs[-paragraph_index].strip() + "\n\n"
            )
            paragraph_index += 1

        if small_paragraph_chunks:
            small_paragraph_chunks.reverse()
            new_chunk += "".join(small_paragraph_chunks)

        new_chunking_style = (
            "multi_paragraph_chunk_with_overlap"
            if paragraph_index > 1
            else "multi_paragraph_chunk_no_overlap"
        )

        return new_chunk, chunk_index + 1, new_chunking_style

    def _process_large_paragraph(
        self,
        chapter_index: int,
        chapter_title: str,
        chunk_index: int,
        current_chunk: str,
        paragraph: str,
        new_chunk_starter_text: str,
    ) -> tuple:
        """Process a large paragraph by splitting into sentences."""
        if os.environ.get("ENV") == "dev":
            print(f"Splitting large paragraph of size {len(paragraph)} into sentences.")

        has_accumulated_content = len(current_chunk) > len(new_chunk_starter_text)

        if has_accumulated_content:
            chunking_style = "sub_chunk_with_paragraph_and_sentence_overlap"
        else:
            chunking_style = "sub_chunk_with_sentence_overlap"

        # Split paragraph into sentences
        sentences = re.split(r"(?<=[.!?]) +", paragraph)
        sentence_index = 0
        overlap_adjusted = False
        end_of_paragraph = False

        while sentence_index < len(sentences):
            sub_chunk = new_chunk_starter_text
            chunk_size = self.target_chunk_size

            if has_accumulated_content:
                chunk_size -= len(current_chunk)
                sub_chunk = current_chunk
                has_accumulated_content = False

            sentences_added = 0
            while len(sub_chunk) < chunk_size:
                sub_chunk += sentences[sentence_index] + " "
                sentence_index += 1
                sentences_added += 1
                if sentence_index >= len(sentences):
                    end_of_paragraph = True
                    break

            if end_of_paragraph:
                sub_chunk += "\n\n"

            if not overlap_adjusted and end_of_paragraph:
                chunking_style = "single_paragraph_chunk_no_overlap"

            self._write_chunk(
                chapter_index, chapter_title, chunk_index, sub_chunk, chunking_style
            )
            chunk_index += 1

            if end_of_paragraph:
                break

            # prevent going back to exact same position creating infinite loop
            previous_sentence_index = sentence_index - sentences_added
            # Move back by overlap amount for next sub-chunk
            sentence_index = max(
                0, previous_sentence_index + 1, sentence_index - self.sentence_overlap
            )
            overlap_adjusted = True

        return new_chunk_starter_text, chunk_index

    def process_book(self, book_data: dict) -> pd.DataFrame:
        """Process all chapters from parsed book data into chunks.

        Args:
            book_data: Dict with 'title', 'author', and 'chapters' list

        Returns:
            DataFrame with chunked content
        """
        self.processed_chunks = []  # Reset for new book

        for chapter in book_data["chapters"]:
            self.chunk_chapter(chapter["index"], chapter["title"], chapter["content"])

        if os.environ.get("ENV") == "dev":
            print(f"Number of chunks: {len(self.processed_chunks)}")

        df = pd.DataFrame(self.processed_chunks)
        df.columns = COLUMN_NAMES
        return df


def get_book_df(
    url: str = None,
    local_filename: str = None,
    target_chunk_size: int = 800,
    sentence_overlap: int = 2,
    small_paragraph_length: int = 200,
    small_paragraph_overlap: int = 2,
    client=None,
) -> dict:
    """Download and process a book into a DataFrame of chunks with embeddings."""
    if url is None or local_filename is None:
        return {
            "status": "error",
            "message": "URL and local filename must be provided.",
        }

    # Download file
    download_result = download_file(url, local_filename)
    if download_result["status"] == "error":
        return download_result

    filepath = download_result["filepath"]
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        book_contents = f.read()

    # Parse based on file extension
    if filepath.endswith(".html"):
        book_data = parse_html_book(book_contents)
    elif filepath.endswith(".txt"):
        book_data = parse_txt_book(book_contents)
    else:
        return {
            "status": "error",
            "message": f"Unsupported file type: {filepath}",
        }

    if not book_data or not book_data.get("chapters"):
        return {
            "status": "error",
            "message": "No chapters found in book. Check parsing logic.",
        }

    # Chunk the parsed book
    chunk_processor = ChunkProcessor(
        target_chunk_size,
        sentence_overlap,
        small_paragraph_length,
        small_paragraph_overlap,
    )
    df = chunk_processor.process_book(book_data)

    if df.empty:
        return {
            "status": "error",
            "message": "No chunks generated. Check chunking logic.",
        }

    # Apply embeddings
    if client is None:
        return {
            "status": "error",
            "message": "GenAI client must be provided.",
        }

    data = apply_embeddings(df, client)
    if data is None:
        return {
            "status": "error",
            "message": "Error applying embeddings.",
        }

    return {
        "status": "success",
        "data": data,
    }
