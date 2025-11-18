"""Module for downloading and processing books into searchable chunks."""

import os
import re
from typing import Union

import pandas as pd
import requests
from google.genai.types import EmbedContentConfig

from .constants import EMBEDDING_MODEL_ID, TEMP_DIR, COLUMN_NAMES

# split with overlap chunking


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

    return response.embeddings[0].values


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


class ChunkProcessor:
    """Class to process book chunks with various chunking strategies."""

    def __init__(
        self,
        local_filename: str,
        target_chunk_size: int,
        sentence_overlap: int,
        small_paragraph_length: int,
        small_paragraph_overlap: int,
    ):
        self.local_filename = local_filename
        self.target_chunk_size = target_chunk_size
        self.sentence_overlap = sentence_overlap
        self.small_paragraph_length = small_paragraph_length
        self.small_paragraph_overlap = small_paragraph_overlap
        self.processed_chunks = []
        self.chunk = ""
        self.full_title = None
        self.chapter_index = None
        self.chunk_index = 0
        self.chunking_style = None

    def download_file(self, url: str) -> dict:
        """Download a file from URL to local workspace.

        Args:
            url: The URL of the text file to download
            local_filename: The filename to save the downloaded content as

        Returns:
            A dict with 'status' and either 'file' path or error 'message'
        """
        if url is None or not url.endswith(".txt"):
            return {
                "status": "error",
                "message": "A valid .txt URL must be provided.",
            }

        # Remove any file extension from local_filename
        self.local_filename = re.sub(r"\.[^./\\]+$", "", self.local_filename)

        filename_ext = f"{self.local_filename}.txt"

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
            return {"status": "success", "file": filepath}
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

    def write_multiparagraph_chunk(self, new_chunk_starter_text: str) -> None:
        """Write the current multi-paragraph chunk and reset state for next chunk."""
        self.processed_chunks.append(
            (
                int(self.chapter_index),
                f"{self.full_title} ({self.chunk_index})",
                len(self.chunk.replace("\n\n", " ").strip()),
                self.chunking_style,
                self.chunk.strip(),
            )
        )

        # Get paragraphs from the chunk we just saved for overlap
        saved_chunk_paragraphs = [p for p in self.chunk.split("\n\n") if p.strip()]
        self.chunk = new_chunk_starter_text

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
            if (
                os.environ.get("ENV") == "dev"
                and paragraph_index > self.small_paragraph_overlap
            ):
                print(
                    f"Reached paragraph overlap limit of {self.small_paragraph_overlap}."
                )

        if small_paragraph_chunks:
            small_paragraph_chunks.reverse()
            self.chunk += "".join(small_paragraph_chunks)

        self.chunking_style = (
            "multi_paragraph_chunk_with_overlap"
            if paragraph_index > 1
            else "multi_paragraph_chunk_no_overlap"
        )
        self.chunk_index += 1

    def read_book_to_chunks(self) -> pd.DataFrame:
        """Read a book file and split it into chunks by chapter.

        Args:
            local_filename: Name of the book file to process
            chunk_size: Maximum size of each text chunk

        Returns:
            DataFrame with columns: chapter_index, title, text
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        filepath = os.path.join(TEMP_DIR, f"{self.local_filename}.txt")
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
                else re.search(
                    r"\*\*\*\s*START OF(.*?)\*\*\*", book, re.IGNORECASE
                ).start()
            )

        if os.environ.get("ENV") == "dev":
            print("Start pos:", start_pos)

        end_match = re.search(r"\*\*\*\s*END OF(.*?)\*\*\*", book, re.IGNORECASE)
        end_pos = end_match.start() if end_match else len(book)

        if os.environ.get("ENV") == "dev":
            print("End pos:", end_pos)

        content = book[start_pos:end_pos]

        # Pattern to match chapter headings with number and name
        # Try two patterns:
        # 1. Title below chapter number: "CHAPTER 1\nThe Title"
        # 2. Title above chapter number: "The Title\n\nCHAPTER 1"

        # Pattern 1: Title below (like "CHAPTER 1\nThe Title")
        chapter_pattern_below = (
            r"^Chapter\s+((?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|"
            r"Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|"
            r"Eighteen|Nineteen|Twenty))\s*\n(.+?)$"
        )

        # Pattern 2: Title above (like "The Title\n\nCHAPTER 1")
        chapter_pattern_above = (
            r"^([A-Z][A-Za-z\s]{2,50})\s*\n+\s*Chapter\s+((?:[IVXLCDM]+|\d+|One|Two|"
            r"Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|"
            r"Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty))\s*$"
        )

        # Try pattern 1 first (title below - most common)
        chapter_matches = list(
            re.finditer(
                chapter_pattern_below, content, flags=re.MULTILINE | re.IGNORECASE
            )
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
            return pd.DataFrame(columns=COLUMN_NAMES)

        # Extract text between chapters
        for chapter_index, match in enumerate(chapter_matches):
            chapter_start = match.start()
            self.chapter_index = chapter_index + 1

            # Extract chapter number and name based on which pattern matched
            if title_position == "above":
                chapter_name = match.group(1).strip()
                chapter_number_raw = match.group(2).strip()
            else:  # below
                chapter_number_raw = match.group(1).strip()
                chapter_name = match.group(2).strip()

            new_chunk_starter_text = (
                f"From Chapter {chapter_number_raw} {chapter_name}: "
            )
            self.full_title = f"Chapter {chapter_number_raw}: {chapter_name}"

            if os.environ.get("ENV") == "dev":
                print(f"Found chapter: {self.full_title}")

            # Get the text until the next chapter (or end of content)
            if self.chapter_index < len(chapter_matches):
                chapter_end = chapter_matches[self.chapter_index].start()
            else:
                chapter_end = len(content)

            # Get the raw chapter text
            chapter_text = content[chapter_start:chapter_end].strip()

            # Remove the chapter heading from the beginning of the text
            # This removes "CHAPTER 1\nA TERRIBLE LOSS\n\n" from the start
            heading_pattern = (
                r"^Chapter\s+(?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|"
                r"Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|"
                r"Eighteen|Nineteen|Twenty)\s*\n.+?\n+"
            )
            chapter_text_without_heading = re.sub(
                heading_pattern,
                "",
                chapter_text,
                count=1,
                flags=re.IGNORECASE,
            ).strip()

            paragraph_chunks = [
                p.replace(
                    "\n", " "
                ).strip()  # replace newlines within paragraphs with spaces
                for p in chapter_text_without_heading.split("\n\n")
                if p.strip()
            ]

            self.chunk = new_chunk_starter_text
            self.chunk_index = 0
            multi_paragraph_chunking_in_progress = False

            for paragraph in paragraph_chunks:

                # If current paragraph is large AND we have accumulated content, save current chunk first
                if (
                    len(paragraph) >= self.target_chunk_size
                    and multi_paragraph_chunking_in_progress
                    and len(self.chunk) > len(new_chunk_starter_text)
                ):
                    self.write_multiparagraph_chunk(new_chunk_starter_text)
                    multi_paragraph_chunking_in_progress = False
                    if os.environ.get("ENV") == "dev":
                        print(
                            "Saved accumulated chunk before processing large paragraph."
                        )

                # Handle large paragraphs that need sentence-level splitting
                if len(paragraph) >= self.target_chunk_size:
                    if os.environ.get("ENV") == "dev":
                        print(
                            f"Splitting large paragraph of size {len(paragraph)} into sentences."
                        )
                    multi_paragraph_chunking_in_progress = False
                    end_of_paragraph = False

                    # Check if we have accumulated content to include in first sub-chunk
                    has_accumulated_content = len(self.chunk) > len(
                        new_chunk_starter_text
                    )
                    if has_accumulated_content:
                        add_previous_paragraphs_to_first_chunk = True
                        self.chunking_style = (
                            "sub_chunk_with_paragraph_and_sentence_overlap"
                        )
                    else:
                        add_previous_paragraphs_to_first_chunk = False
                        self.chunking_style = "sub_chunk_with_sentence_overlap"

                    # Split paragraph into sentences
                    sentences = re.split(
                        r"(?<=[.!?]) +", paragraph
                    )  # maybe do better sentence splitting later
                    sentence_index = 0
                    overlap_adjusted = False
                    while sentence_index < len(sentences):
                        sub_chunk = new_chunk_starter_text
                        chunk_size = self.target_chunk_size

                        if add_previous_paragraphs_to_first_chunk:
                            chunk_size -= len(
                                self.chunk
                            )  # adjust for existing chunk content
                            sub_chunk = self.chunk
                            add_previous_paragraphs_to_first_chunk = False

                        while len(sub_chunk) < chunk_size:
                            sub_chunk += sentences[sentence_index] + " "
                            sentence_index += 1
                            if sentence_index >= len(sentences):
                                # Indicate end of paragraph
                                end_of_paragraph = True
                                break

                        if end_of_paragraph:
                            sub_chunk += "\n\n"

                        if not overlap_adjusted and end_of_paragraph:
                            self.chunking_style = "single_paragraph_chunk_no_overlap"

                        self.processed_chunks.append(
                            (
                                int(self.chapter_index),
                                f"{self.full_title} ({self.chunk_index})",
                                len(sub_chunk.replace("\n\n", " ").strip()),
                                self.chunking_style,
                                sub_chunk.strip(),
                            )
                        )
                        self.chunk_index += 1

                        if end_of_paragraph:
                            break

                        # Move back by overlap amount for next sub-chunk
                        sentence_index = max(0, sentence_index - self.sentence_overlap)
                        overlap_adjusted = True

                    self.chunk = new_chunk_starter_text

                else:
                    # Small paragraph - add to accumulating chunk
                    multi_paragraph_chunking_in_progress = True
                    if self.chunking_style != "multi_paragraph_chunk_with_overlap":
                        self.chunking_style = "multi_paragraph_chunk_no_overlap"
                    self.chunk += paragraph + "\n\n"

                    # Check if accumulated chunk has reached target size
                    if (
                        len(self.chunk.replace("\n\n", " ").strip())
                        >= self.target_chunk_size
                    ):
                        self.write_multiparagraph_chunk(new_chunk_starter_text)

            # Save any remaining chunk content at end of chapter
            if len(self.chunk) > len(new_chunk_starter_text):
                self.processed_chunks.append(
                    (
                        int(self.chapter_index),
                        f"{self.full_title} ({self.chunk_index})",
                        len(self.chunk.replace("\n\n", " ").strip()),
                        self.chunking_style,
                        self.chunk.strip(),
                    )
                )

        # Add introduction if it exists before first chapter
        if chapter_matches and intro_matches:
            intro_text = (
                content[: chapter_matches[0].start()]
                .replace("Introduction", "")
                .replace("\n", " ")
                .strip()
            )
            # TODO: do the same chunking for intro as for chapters
            if len(intro_text) > 100:
                self.processed_chunks.insert(
                    0, (0, "Introduction", len(intro_text), None, intro_text)
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
) -> Union[pd.DataFrame, dict]:
    """Download and process a book into a DataFrame of chunks with embeddings."""
    if url is None or local_filename is None:
        return {
            "status": "error",
            "message": "URL and local filename must be provided.",
        }

    chunk_processor = ChunkProcessor(
        local_filename,
        target_chunk_size,
        sentence_overlap,
        small_paragraph_length,
        small_paragraph_overlap,
    )
    response = chunk_processor.download_file(url)

    if response["status"] == "error":
        return response
    if client is None:
        return {
            "status": "error",
            "message": "GenAI client must be provided.",
        }

    data = apply_embeddings(
        chunk_processor.read_book_to_chunks(),
        client,
    )
    if data is None:
        return {
            "status": "error",
            "message": "No data to process after reading book. Check parsing logic.",
        }
    return {
        "status": "success",
        "data": data,
    }
