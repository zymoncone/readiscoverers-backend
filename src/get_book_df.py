"""Module for downloading and processing books into searchable chunks."""

import glob
import hashlib
import os
import re
import json
from typing import Union
from urllib.parse import parse_qsl, urlparse, urlunparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google.genai.types import EmbedContentConfig

from .constants import (
    EMBEDDING_MODEL_ID,
    TEMP_DIR,
    PROCESSED_BOOKS_EMBEDDINGS_DIR,
    PROCESSED_BOOKS_METADATA_DIR,
    USE_GCS,
    GCS_BUCKET_NAME,
    GCS_EMBEDDINGS_PREFIX,
    GCS_METADATA_PREFIX,
    COLUMN_NAMES,
    LARGE_PARAGRAPH_TOLERANCE,
)
from .gcs_utils import (
    read_pickle_with_cache,
    read_json_with_cache,
    blob_exists_in_gcs,
)
from .parse_html import parse_html_book
from .parse_txt import parse_txt_book


def is_book_cached(book_title: str) -> bool:
    """Check if a book has already been processed and cached.

    Args:
        book_title: The title of the book to check

    Returns:
        True if both the pickle file and metadata exist, False otherwise
    """
    if not book_title:
        return False

    filename = book_title.replace(" ", "_").lower()

    if USE_GCS:
        # Check GCS
        embeddings_blob = f"{GCS_EMBEDDINGS_PREFIX}/{filename}.pkl"
        metadata_blob = f"{GCS_METADATA_PREFIX}/{filename}_metadata.json"
        return blob_exists_in_gcs(
            GCS_BUCKET_NAME, embeddings_blob
        ) and blob_exists_in_gcs(GCS_BUCKET_NAME, metadata_blob)
    else:
        # Check local filesystem
        pickle_path = f"{PROCESSED_BOOKS_EMBEDDINGS_DIR}/{filename}.pkl"
        metadata_path = f"{PROCESSED_BOOKS_METADATA_DIR}/{filename}_metadata.json"
        return os.path.exists(pickle_path) and os.path.exists(metadata_path)


def load_cached_book(
    book_title: str,
) -> Union[dict, None]:
    """Load a cached book's dataframe and metadata.

    Args:
        book_title: The title of the book to load

    Returns:
        Dict with status, book_data, book_title, book_author, and filename, or error dict
    """
    if not book_title:
        return None

    filename = book_title.replace(" ", "_").lower()

    try:
        if USE_GCS:
            # Load from GCS with Redis caching
            embeddings_blob = f"{GCS_EMBEDDINGS_PREFIX}/{filename}.pkl"
            metadata_blob = f"{GCS_METADATA_PREFIX}/{filename}_metadata.json"

            df = read_pickle_with_cache(GCS_BUCKET_NAME, embeddings_blob)
            metadata = read_json_with_cache(GCS_BUCKET_NAME, metadata_blob)

            if df is None or metadata is None:
                return None
        else:
            # Load from local filesystem
            pickle_path = f"{PROCESSED_BOOKS_EMBEDDINGS_DIR}/{filename}.pkl"
            metadata_path = f"{PROCESSED_BOOKS_METADATA_DIR}/{filename}_metadata.json"

            if not os.path.exists(pickle_path) or not os.path.exists(metadata_path):
                return None

            df = pd.read_pickle(pickle_path)

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return {
            "status": "success",
            "book_data": df,
            "book_title": metadata.get("book_title"),
            "book_author": metadata.get("book_author"),
            "filename": filename,
            "cached": True,
        }
    except Exception as e:
        if os.environ.get("ENV") == "dev":
            print(f"Error loading cached book {book_title}: {str(e)}")
        return None


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


def _normalize_url(url: str) -> str:
    """Normalize URLs to improve cache reuse across equivalent links."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.hostname.lower() if parsed.hostname else ""
    if parsed.port and parsed.port not in (80, 443):
        netloc = f"{netloc}:{parsed.port}"

    query_params = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    normalized_query = "&".join([f"{k}={v}" for k, v in query_params])

    normalized = urlunparse(
        (
            scheme,
            netloc,
            parsed.path or "",
            parsed.params or "",
            normalized_query,
            "",
        )
    )
    return normalized


def _get_cached_filepath(url: str) -> str:
    """Return a stable cache filepath derived from the normalized URL."""
    normalized_url = _normalize_url(url)
    file_extension = ".html" if normalized_url.endswith(".html") else ".txt"
    url_hash = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(TEMP_DIR, "download_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{url_hash}{file_extension}")


def _find_cached_filepath(url: str) -> str | None:
    """Find a cached download, including legacy files with a UUID suffix."""
    filepath = _get_cached_filepath(url)
    if os.path.exists(filepath):
        return filepath

    cache_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    prefix, extension = os.path.splitext(filename)
    legacy_pattern = os.path.join(cache_dir, f"{prefix}_*{extension}")
    legacy_matches = glob.glob(legacy_pattern)
    if legacy_matches:
        legacy_path = legacy_matches[0]
        # Migrate legacy file to new path
        try:
            os.rename(legacy_path, filepath)
            if os.environ.get("ENV") == "dev":
                print(f"Migrated legacy cache: {legacy_path} -> {filepath}")
        except OSError:
            pass  # If rename fails, just use the legacy path
        return filepath

    return None


def _create_requests_session(
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: tuple = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Create a requests session configured for retries and resiliency."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "ReadiscoverersDownloader/1.0",
        }
    )
    return session


def download_file(url: str, local_filename: str) -> dict:
    """Download a file from URL to local workspace, caching by URL.

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

    # Ensure temp directory exists and is a directory
    os.makedirs(TEMP_DIR, exist_ok=True)

    cached_filepath = _find_cached_filepath(url)
    if cached_filepath:
        if os.environ.get("ENV") == "dev":
            print(
                f"Using cached download for URL: {url}\n"
                f"Cache path: {cached_filepath}"
            )
        return {"status": "success", "filepath": cached_filepath, "cached": True}

    filepath = _get_cached_filepath(url)
    temp_filepath = f"{filepath}.part"
    session = _create_requests_session()

    try:
        with session.get(url, timeout=(10, 60), stream=True) as response:
            response.raise_for_status()
            with open(temp_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        os.replace(temp_filepath, filepath)
        if os.environ.get("ENV") == "dev":
            print(f"Downloaded {filepath}")
        return {"status": "success", "filepath": filepath, "cached": False}
    except requests.RequestException as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return {
            "status": "error",
            "message": f"Error downloading {url}: {str(e)}",
        }
    except IOError as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return {
            "status": "error",
            "message": f"Error writing file {filepath}: {str(e)}",
        }
    finally:
        session.close()


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
        self, chapter_index: int, chapter_title: str, content: str, book_title: str
    ) -> None:
        """Process a single chapter's content into chunks.

        Args:
            chapter_index: The index of the chapter
            chapter_title: The full title of the chapter
            content: The paragraph content joined with \\n\\n
            book_title: The title of the book
        """
        # Split content back into paragraphs
        if os.environ.get("ENV") == "dev":
            print(f"Chunking Chapter {chapter_index}: '{chapter_title}'")
        paragraph_chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

        new_chunk_starter_text = (
            f"Book: {book_title}, Chapter: {chapter_index} {chapter_title} - "
        )
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

        max_acceptable_size = self.target_chunk_size * LARGE_PARAGRAPH_TOLERANCE

        # If paragraph is close enough to target size, keep it whole.
        # This prevents one sentence chunks with no context
        if len(paragraph) <= max_acceptable_size:
            if os.environ.get("ENV") == "dev":
                print(
                    f"Paragraph size {len(paragraph)} within tolerance ({max_acceptable_size}), keeping whole."
                )

            # Write the whole paragraph as a chunk
            chunk_to_write = (
                current_chunk + paragraph + "\n\n"
                if has_accumulated_content
                else new_chunk_starter_text + paragraph + "\n\n"
            )
            chunking_style = "single_paragraph_chunk_no_overlap"

            self._write_chunk(
                chapter_index,
                chapter_title,
                chunk_index,
                chunk_to_write,
                chunking_style,
            )
            return new_chunk_starter_text, chunk_index + 1

        # Paragraph is too large, split into smaller chunks
        if has_accumulated_content:
            chunking_style = "sub_chunk_with_paragraph_and_sentence_overlap"
        else:
            chunking_style = "sub_chunk_with_sentence_overlap"

        # Split paragraph into sentences
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
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
            self.chunk_chapter(
                chapter["index"],
                chapter["title"],
                chapter["content"],
                book_data["title"],
            )

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
    """Download and process a book into a DataFrame of chunks with embeddings.

    Checks cache first after parsing to see if the book has already been processed.
    If cached, returns the stored data without re-embedding.
    """
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

    # Extract book metadata early
    book_title = book_data.get("title")
    book_author = book_data.get("author")

    # Check if this book is already cached (avoids expensive re-embedding)
    if book_title and is_book_cached(book_title):
        if os.environ.get("ENV") == "dev":
            print(f"Book '{book_title}' found in cache, skipping reprocessing.")

        cached_result = load_cached_book(book_title)
        if cached_result:
            return cached_result

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

    processed_data = apply_embeddings(df, client)
    if processed_data is None:
        return {
            "status": "error",
            "message": "Error applying embeddings.",
        }

    filename = book_title.replace(" ", "_").lower() if book_title else local_filename

    return {
        "status": "success",
        "book_title": book_title,
        "book_author": book_author,
        "filename": filename,
        "book_data": processed_data,
        "cached": False,
    }
