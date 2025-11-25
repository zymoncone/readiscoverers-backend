"""Constants used throughout the application."""

import re

EMBEDDING_MODEL_ID = "text-embedding-005"
TEMP_DIR = "/tmp"
COLUMN_NAMES = [
    "chapter_index",
    "title",
    "chunk_length",
    "chunking_style",
    "text",
]

CHAPTER_NUMBERS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"twenty[- ]one|twenty[- ]two|twenty[- ]three|twenty[- ]four|twenty[- ]five|"
    r"twenty[- ]six|twenty[- ]seven|twenty[- ]eight|twenty[- ]nine|thirty|"
    r"thirty[- ]one|thirty[- ]two|thirty[- ]three|thirty[- ]four|thirty[- ]five|"
    r"thirty[- ]six|thirty[- ]seven|thirty[- ]eight|thirty[- ]nine|forty"
    # Up to forty, can be extended as needed
)

# "CHAPTER [number] [Title Text]" - number followed by text
CHAPTER_WITH_TITLE_PATTERN = re.compile(
    r"^chapter\s+(?P<number>[ivxlcdm]+|\d+|" + CHAPTER_NUMBERS + r")\s+(?P<title>.+)$",
    re.IGNORECASE,
)

# "[Title Text]--CHAPTER [number]" - title comes before chapter number
TITLE_BEFORE_CHAPTER_PATTERN = re.compile(
    r"^(?P<title>.+?)\s*[-–—]*\s*chapter\s+(?:[ivxlcdm]+|\d+|"
    + CHAPTER_NUMBERS
    + r")$",
    re.IGNORECASE,
)

# Page number tag pattern, i.e., [page 12] or [pg. 12]
PAGE_NUMBER_TAG_PATTERN = re.compile(r"\[?\s*(?:page|pg)\.?\s*\d+\s*\]?", re.IGNORECASE)

# "CHAPTER [number]" only - look for subtitle in siblingxw
CHAPTER_ONLY_PATTERN = re.compile(
    r"^chapter\s+(?:[ivxlcdm]+|\d+|" + CHAPTER_NUMBERS + r")[\s.:]*$", re.IGNORECASE
)
