import os
import re
import sys

import pandas as pd
from bs4 import BeautifulSoup

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from src.parse_html import get_paragraph_with_dropcap


def normalize_text(text):
    # Handle NaN or None inputs
    if pd.isna(text) or text is None:
        return ""

    if type(text) != str:
        raise ValueError(f"Input text must be a string for text: {text}")

    # Check if text contains HTML tags
    if "<p>" in text or "<" in text:
        text = f"<p>{text}"  # adding missing <p> tag at the start
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

    # Remove quotes
    text = re.sub(r'["‘”’“\']', "", text)
    # Normalize whitespace
    return re.sub(r"\s+", " ", text.lower().strip())


def remove_chapter_chunk_tag(text):
    text_match = re.search(r"From Chapter\s+.+?:\s*(.+)", text, re.DOTALL)
    if text_match:
        return text_match.group(1).strip()
    return text


def find_chunk_locations_with_continuity(filepath, expected_text):
    """
    Find chunks that contain the expected text with continuity across multiple chunks.
    Uses word-by-word matching to handle text that spans chunk boundaries.
    Returns list of chunk indices where the text spans, or empty list if not found.
    """
    temp_df = pd.read_pickle(filepath)

    # Normalize and split expected text into words
    normalized_expected = normalize_text(expected_text)
    expected_words = normalized_expected.split()

    if not expected_words:
        print("WARNING: No words found in expected text")
        return []

    print(f"Looking for {len(expected_words)} words from expected text")

    # Find the chunk containing the first few words
    first_phrase = " ".join(
        expected_words[: min(5, len(expected_words))]
    )  # Use first 5 words
    start_chunk_idx = None

    for chunk_idx, row in temp_df.iterrows():
        chunk_text = normalize_text(remove_chapter_chunk_tag(row["text"]))
        if first_phrase in chunk_text:
            start_chunk_idx = chunk_idx
            break

    if start_chunk_idx is None:
        print(f"WARNING: First phrase not found: '{first_phrase}'")
        return []

    # Now verify continuity word by word from start_chunk_idx
    matched_chunks = [start_chunk_idx]
    current_chunk_idx = start_chunk_idx
    word_idx = 0

    while word_idx < len(expected_words):
        if current_chunk_idx >= len(temp_df):
            print(
                f"WARNING: Ran out of chunks at word {word_idx}/{len(expected_words)}"
            )
            return []

        current_text = normalize_text(
            remove_chapter_chunk_tag(temp_df.iloc[current_chunk_idx]["text"])
        )
        chunk_words = current_text.split()

        # Find where we are in the current chunk
        words_matched_in_chunk = 0

        # Try to match consecutive words from expected_words starting at word_idx
        for i in range(word_idx, len(expected_words)):
            # Look for the current expected word in remaining chunk words
            expected_word = expected_words[i]

            # Create a sliding window of chunk text to find the word sequence
            remaining_chunk_text = " ".join(chunk_words[words_matched_in_chunk:])

            if expected_word in remaining_chunk_text:
                # Find the position and advance
                word_position = (
                    remaining_chunk_text.split().index(expected_word)
                    if expected_word in remaining_chunk_text.split()
                    else -1
                )
                if word_position >= 0:
                    words_matched_in_chunk += word_position + 1
                    word_idx = i + 1
                else:
                    # Word is part of a larger match, just continue
                    word_idx = i + 1
            else:
                # Word not found in remaining chunk text
                break

        if word_idx == 0 or words_matched_in_chunk == 0:
            # Couldn't match any words in this chunk - continuity broken
            print(
                f"WARNING: Continuity broken at word {word_idx}: '{expected_words[word_idx] if word_idx < len(expected_words) else 'END'}'"
            )
            print(f"Current chunk text: {current_text}")
            print(f"For filepath: {filepath}, chunk index: {current_chunk_idx}")

            return []

        # If there are more words to find, move to next chunk
        if word_idx < len(expected_words):
            current_chunk_idx += 1
            if (
                current_chunk_idx < len(temp_df)
                and current_chunk_idx not in matched_chunks
            ):
                matched_chunks.append(current_chunk_idx)

    # Successfully found all words with continuity
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

        if use_expected_settings:
            data.append(
                {
                    "expected_chunk_index": chunk_index,
                    "expected_chapter_number": chapter_number,
                    "expected_chapter_title": chapter_title,
                    "expected_chunk_in_chapter_index": chunk_in_chapter_index,
                    "all_expected_chunks": all_chunks,
                    "expected_filename": matched_filename,
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
                }
            )

    # Combine with previous dataframe
    df = pd.DataFrame(data)
    return pd.concat([selected_questions_df.reset_index(drop=True), df], axis=1)
