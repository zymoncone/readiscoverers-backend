import os
import re
import sys

import pandas as pd
from bs4 import BeautifulSoup

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from src.parse_html import get_paragraph_with_dropcap
from src.constants import CHAPTER_NUMBERS


def normalize_text(text, lowercase: bool = True) -> str:
    # Handle NaN or None inputs
    if pd.isna(text) or text is None:
        return ""

    if type(text) != str:
        raise ValueError(f"Input text must be a string for text: {text}")

    # Check if text contains HTML tags
    if "<p>" in text or "<" in text:
        text = text.lstrip(">")  # removing leading '>' if present
        text = f"<p>{text}</p>"  # adding missing <p></p> tags at the start
        soup = BeautifulSoup(text, "html.parser")

        # Get all paragraph elements
        p_elements = soup.find_all("p")

        if p_elements:
            # Extract text using the same logic as parse_html.py
            paragraphs = []
            for p in p_elements:
                p_text = get_paragraph_with_dropcap(p)
                if p_text and p_text.lower() != "view image":
                    paragraphs.append(p_text)

            text = " ".join(paragraphs) if paragraphs else soup.get_text(strip=True)
        else:
            # No p tags, just get text
            text = soup.get_text(strip=True)

    # Remove page number patterns like [pg 123], [page 123], [123], « 123 »
    text = re.sub(
        r"\[?\s*(?:pg|page)\.?\s*\d+\s*\]?|«\s*\d+\s*»|\[\d+\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove chapter markers at the end like "chapter 20." or "chapter twenty"
    text = re.sub(
        rf"\s*chapter\s+(?:\d+|[ivxlcdm]+|{CHAPTER_NUMBERS})\s*\.?\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove quotes
    text = re.sub(r'["‘”’“\']', "", text)
    # Normalize ellipsis (both Unicode … and three dots ...)
    text = re.sub(r"…|\.\.\.", "", text)
    # Replace em dashes and other dashes with spaces to prevent word merging
    text = re.sub(r"[—–-]", " ", text)
    # Replace commas, found missing
    text = re.sub(r",", "", text)
    # # Remove all other punctuation
    # text = re.sub(r"[.,;:!?]", "", text)
    if lowercase:
        text = text.lower()
    # Normalize whitespace
    return re.sub(r"\s+", " ", text.strip())


def remove_chapter_chunk_tag(text):
    text_match = re.search(
        r"Book:\s+[^,]+,\s+Chapter:\s+\d+\s+[^-]+-\s*(.+)", text, re.DOTALL
    )
    if text_match:
        return text_match.group(1).strip()
    return text


def find_chunk_locations_with_continuity(filepath, expected_text):
    """
    Find chunks that contain the expected text with continuity across multiple chunks.
    Matches sentence-by-sentence, skipping duplicate sentences from previous chunks.
    Returns list of chunk indices where the text spans, or empty list if not found.
    """
    temp_df = pd.read_pickle(filepath)

    def split_sentences(text):
        """Split text into sentences. Normalizes first, then splits."""
        # Normalize first to handle HTML, quotes, etc. consistently
        normalized = normalize_text(text, lowercase=False)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?:])\s+(?=[A-Z"])', normalized)

        # Filter out empty strings
        return [s.strip().lower() for s in sentences if s.strip()]

    # Split expected text into sentences (already normalized)
    expected_sentences = split_sentences(expected_text)

    if not expected_sentences:
        print("WARNING: No sentences found in expected text")
        return []

    print(f"Looking for {len(expected_sentences)} sentences from expected text")
    print(f"First expected sentence: '{expected_sentences[0]}")

    # Find the chunk containing the first sentence
    first_sentence = expected_sentences[0]
    start_chunk_idx = None

    print(f"Searching for first sentence in chunks...")

    # First pass: find starting chunk
    for chunk_idx, row in temp_df.iterrows():
        chunk_text = remove_chapter_chunk_tag(row["text"])
        chunk_sentences = split_sentences(chunk_text)

        if first_sentence in chunk_sentences:
            start_chunk_idx = chunk_idx
            print(f"Found first sentence in chunk {chunk_idx}")
            break

    if start_chunk_idx is None:
        print(f"WARNING: First sentence not found in any chunk")
        return []

    # Second pass: verify continuity sentence-by-sentence
    matched_chunks = [start_chunk_idx]
    current_chunk_idx = start_chunk_idx
    sentence_idx = 0  # Which expected sentence we're looking for
    seen_sentences = set()  # Track sentences we've already matched

    while sentence_idx < len(expected_sentences):
        if current_chunk_idx >= len(temp_df):
            print(
                f"WARNING: Ran out of chunks at sentence {sentence_idx}/{len(expected_sentences)}"
            )
            return []

        # Get current chunk's sentences (already normalized from split_sentences)
        raw_chunk_text = remove_chapter_chunk_tag(
            temp_df.iloc[current_chunk_idx]["text"]
        )
        chunk_sentences = split_sentences(raw_chunk_text)

        # Filter out sentences we've already seen (duplicates from overlap)
        is_first_chunk = current_chunk_idx == start_chunk_idx
        if not is_first_chunk:
            # Skip sentences that were in previous chunks
            chunk_sentences = [s for s in chunk_sentences if s not in seen_sentences]

        # Add current chunk's sentences to seen set
        seen_sentences.update(chunk_sentences)

        if is_first_chunk:
            print(f"First chunk {current_chunk_idx} - {len(chunk_sentences)} sentences")
        else:
            print(
                f"Chunk {current_chunk_idx} - {len(chunk_sentences)} new sentences after dedup"
            )

        # Try to match expected sentences in this chunk
        chunk_sentence_idx = 0
        matched_in_this_chunk = False

        while sentence_idx < len(expected_sentences) and chunk_sentence_idx < len(
            chunk_sentences
        ):
            expected_sent = expected_sentences[sentence_idx]
            chunk_sent = chunk_sentences[chunk_sentence_idx]

            if expected_sent == chunk_sent:
                # Match! Move to next expected sentence
                sentence_idx += 1
                chunk_sentence_idx += 1
                matched_in_this_chunk = True
                print(f"  ✓ Matched sentence {sentence_idx}/{len(expected_sentences)}")
            else:
                # No match - check if we've already matched something in this chunk
                if matched_in_this_chunk:
                    # We matched earlier sentences but now it broke - continuity error
                    print(f"WARNING: Continuity broken at sentence {sentence_idx}")
                    print(f"  Expected: '{expected_sent}'")
                    print(f"  Got: '{chunk_sent}'")
                    return []
                else:
                    # Haven't matched anything in this chunk yet, try next sentence in chunk
                    chunk_sentence_idx += 1

        # If we haven't matched anything in this chunk and it's not the first, something's wrong
        if not matched_in_this_chunk and not is_first_chunk:
            print(f"WARNING: No matches found in chunk {current_chunk_idx}")
            print(
                f"  Still looking for sentence {sentence_idx}: '{expected_sentences[sentence_idx][:100]}...'"
            )
            return []

        # If we still have sentences to match, move to next chunk
        if sentence_idx < len(expected_sentences):
            current_chunk_idx += 1
            if (
                current_chunk_idx < len(temp_df)
                and current_chunk_idx not in matched_chunks
            ):
                matched_chunks.append(current_chunk_idx)

    # Successfully found all sentences with continuity
    print(
        f"SUCCESS: Found text spanning {len(matched_chunks)} chunk(s): {matched_chunks}"
    )
    return matched_chunks


def find_chunk_location_from_text(filepath, expected_text):
    matched_chunks = find_chunk_locations_with_continuity(filepath, expected_text)

    if not matched_chunks:
        return (None, None, None, None, [])

    # Get metadata from the first matched chunk
    temp_df = pd.read_pickle(filepath)
    primary_chunk_idx = matched_chunks[0]
    row = temp_df.iloc[primary_chunk_idx]

    chapter_number = int(row["chapter_index"])
    chapter_title = str(row["title"])
    chunk_in_chapter_match = re.search(r"(.*)\((\d+)\)", chapter_title)
    if chunk_in_chapter_match and chunk_in_chapter_match.group(2).isdigit():
        chunk_in_chapter_index = int(chunk_in_chapter_match.group(2))
        chapter_title = chunk_in_chapter_match.group(1).strip()
    else:
        chunk_in_chapter_index = None

    return (
        primary_chunk_idx,
        chapter_number,
        chapter_title,
        chunk_in_chapter_index,
        matched_chunks,
    )


def precompute_text_locations_in_chunks(
    selected_questions_df: pd.DataFrame, use_expected_settings: bool = True
):
    data = []

    text_col = "Best Answer" if use_expected_settings else "excerpt"

    for idx, question_row in selected_questions_df.iterrows():

        if pd.isna(question_row.get(text_col)) or not question_row.get(text_col):
            print(
                f"WARNING: Skipping row {idx} - empty/NaN text in '{text_col}' column"
            )
            print(f"  Question: {question_row.get('question', 'N/A')}")

            # Return None value for the missing row
            if use_expected_settings:
                data.append(
                    {
                        "expected_chunk_index": None,
                        "expected_chapter_number": None,
                        "expected_chapter_title": None,
                        "expected_chunk_in_chapter_index": None,
                        "all_expected_chunks": [],
                        "expected_filename": None,
                        "excerpt_cleaned": "",
                    }
                )
            else:
                data.append(
                    {
                        "model_chunk_index": None,
                        "model_chapter_number": None,
                        "model_chapter_title": None,
                        "model_chunk_in_chapter_index": None,
                        "all_model_chunks": [],
                        "model_filename": None,
                        "excerpt_cleaned": "",
                    }
                )
            continue  # Skip to next row

        if use_expected_settings:
            filename = question_row["Book Title"].lower().replace(" ", "_").strip()
            book_filepath = f"../temp/{filename}.pkl"

            (
                chunk_index,
                chapter_number,
                chapter_title,
                chunk_in_chapter_index,
                all_chunks,
            ) = (
                find_chunk_location_from_text(book_filepath, question_row[text_col])
                if os.path.exists(book_filepath)
                else (None, None, None, None, [])
            )

            matched_filename = filename
        else:
            # When not using expected settings, search through all available pickle files
            chunk_index = None
            chapter_number = None
            chapter_title = None
            chunk_in_chapter_index = None
            all_chunks = []
            matched_filename = None

            # Get all pickle files in ../temp/
            import glob

            pickle_files = glob.glob("../temp/*.pkl")

            for pickle_path in pickle_files:
                result = find_chunk_location_from_text(
                    pickle_path, question_row[text_col]
                )
                (
                    chunk_index,
                    chapter_number,
                    chapter_title,
                    chunk_in_chapter_index,
                    all_chunks,
                ) = result

                if chunk_index is not None:
                    # Found a match!
                    matched_filename = os.path.basename(pickle_path).replace(".pkl", "")
                    break

        if not matched_filename:
            print(
                f"WARNING: No matching chunk found for question: {question_row['question']}"
            )

        cleaned_text = (
            normalize_text(question_row[text_col])
            if not pd.isna(question_row.get(text_col))
            else ""
        )

        if use_expected_settings:
            data.append(
                {
                    "expected_chunk_index": chunk_index,
                    "expected_chapter_number": chapter_number,
                    "expected_chapter_title": chapter_title,
                    "expected_chunk_in_chapter_index": chunk_in_chapter_index,
                    "all_expected_chunks": all_chunks,
                    "expected_filename": matched_filename,
                    "excerpt_cleaned": cleaned_text,
                }
            )
        else:
            data.append(
                {
                    "model_chunk_index": chunk_index,
                    "model_chapter_number": chapter_number,
                    "model_chapter_title": chapter_title,
                    "model_chunk_in_chapter_index": chunk_in_chapter_index,
                    "all_model_chunks": all_chunks,
                    "model_filename": matched_filename,
                    "excerpt_cleaned": cleaned_text,
                }
            )

    # Combine with previous dataframe
    df = pd.DataFrame(data)
    return pd.concat([selected_questions_df.reset_index(drop=True), df], axis=1)
