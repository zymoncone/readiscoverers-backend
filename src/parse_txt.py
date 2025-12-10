"""parse .txt book content into structured chapters and paragraphs."""

import os
import re
from typing import Union


def parse_txt_book(text_file: str) -> Union[dict, None]:
    """Parses a plain text book file into chapters.

    The extent of this function is not robust parsing for all possible
    formats, but rather to handle the common formats found in Project
    Gutenberg plain text files.

    The html parser (./parse_html.py) is more robust and should be
    preferred when available.
    """
    # Find the start of the actual content
    intro_matches = list(re.finditer(r"Introduction", text_file))

    if len(intro_matches) >= 2:
        start_pos = intro_matches[1].start()
    else:
        start_pos = (
            intro_matches[0].start()
            if intro_matches
            else re.search(
                r"\*\*\*\s*START OF(.*?)\*\*\*", text_file, re.IGNORECASE
            ).start()
        )

    if os.environ.get("ENV") == "dev":
        print("Start pos:", start_pos)

    end_match = re.search(r"\*\*\*\s*END OF(.*?)\*\*\*", text_file, re.IGNORECASE)
    end_pos = end_match.start() if end_match else len(text_file)

    if os.environ.get("ENV") == "dev":
        print("End pos:", end_pos)

    content = text_file[start_pos:end_pos]

    # Pattern to match chapter headings with number and name
    # Try two patterns:
    # 1. Title below chapter number: "CHAPTER 1\nThe Title"
    # 2. Title above chapter number: "The Title\n\nCHAPTER 1"

    # Pattern 1: Title below (like "CHAPTER 1\nThe Title" or "Introduction" standalone)
    chapter_pattern_below = (
        r"^(?:Chapter\s+((?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|"
        r"Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|"
        r"Eighteen|Nineteen|Twenty))\s*\n(.+?)|(Introduction|Prologue))$"
    )

    # Pattern 2: Title above (like "The Title\n\nCHAPTER 1")
    chapter_pattern_above = (
        r"^([A-Z][A-Za-z\s]{2,50})\s*\n+\s*Chapter\s+((?:[IVXLCDM]+|\d+|One|Two|"
        r"Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|"
        r"Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty))\s*$"
    )

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
        return None

    chapters = []
    # Extract text between chapters
    for chapter_index, match in enumerate(chapter_matches):
        chapter_start = match.start()

        # Extract chapter number and name based on which pattern matched
        if title_position == "above":
            chapter_name = match.group(1).strip()
        else:  # below
            if match.group(3):  # Introduction or Prologue matched
                chapter_name = match.group(3).strip()
            else:  # Regular chapter matched
                chapter_name = match.group(2).strip()

        # Get the text until the next chapter (or end of content)
        if chapter_index < len(chapter_matches) - 1:
            chapter_end = chapter_matches[chapter_index + 1].start()
        else:
            chapter_end = len(content)

        # Get the raw chapter text
        chapter_text = content[chapter_start:chapter_end].strip()

        # Remove the chapter heading from the beginning of the text
        heading_pattern = (
            r"^(?:Chapter\s+(?:[IVXLCDM]+|\d+|One|Two|Three|Four|Five|Six|Seven|Eight|"
            r"Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|"
            r"Eighteen|Nineteen|Twenty)\s*\n.+?|Introduction|Prologue)\n+"
        )
        chapter_text_without_heading = re.sub(
            heading_pattern,
            "",
            chapter_text,
            count=1,
            flags=re.IGNORECASE,
        ).strip()

        chapter_text_cleaned = [
            p.replace("\n", " ").strip()
            for p in chapter_text_without_heading.split("\n\n")
            if p.strip()
        ]

        if chapter_name == "Introduction" or chapter_name == "Prologue":
            chapter_index_adjusted = 0
        else:
            chapter_index_adjusted = chapter_index

        chapters.append(
            {
                "index": chapter_index_adjusted,
                "title": chapter_name,
                "content": "\n\n".join(chapter_text_cleaned),
                "paragraph_count": len(chapter_text_cleaned),
            }
        )

    return {
        "title": "Unknown Title",
        "author": "Unknown Author",
        "chapters": chapters,
    }
