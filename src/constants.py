"""Constants used throughout the application."""

import os
import re

EMBEDDING_MODEL_ID = "text-embedding-005"
MODEL_ID = "gemini-2.0-flash-001"

# Temporary directory for downloads and dev files (cleaned up regularly)
TEMP_DIR = "/tmp"

# Google Cloud Storage configuration
USE_GCS = os.environ.get("USE_GCS", "false").lower() == "true"
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "processed-books")
GCS_EMBEDDINGS_PREFIX = "embeddings"
GCS_METADATA_PREFIX = "metadata"

# Redis configuration for caching
USE_REDIS = os.environ.get("USE_REDIS", "false").lower() == "true"
REDIS_URL = os.environ.get("REDIS_URL", None)

if USE_REDIS and not REDIS_URL:
    raise ValueError(
        "USE_REDIS is true but REDIS_URL is not set in environment variables."
    )


REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(
    os.environ.get("REDIS_TTL_SECONDS", "86400")
)  # 24 hours default

# Local persistent directory structure (used when USE_GCS=false)
PROCESSED_BOOKS_DIR = os.environ.get("PROCESSED_BOOKS_DIR", "./processed_books")
PROCESSED_BOOKS_EMBEDDINGS_DIR = os.path.join(PROCESSED_BOOKS_DIR, "embeddings")
PROCESSED_BOOKS_METADATA_DIR = os.path.join(PROCESSED_BOOKS_DIR, "metadata")
COLUMN_NAMES = [
    "chapter_index",
    "title",
    "chunk_length",
    "chunking_style",
    "text",
]
LARGE_PARAGRAPH_TOLERANCE = 1.5  # 50% tolerance for large paragraphs

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

# Page number tag pattern, i.e., [page 12], [pg. 12], [12], or « 12 »
PAGE_NUMBER_TAG_PATTERN = re.compile(
    r"\[?\s*(?:(?:page|pg)\.?\s*)?\d+\s*\]?|«\s*\d+\s*»", re.IGNORECASE
)

# "CHAPTER [number]" only - look for subtitle in siblingxw
CHAPTER_ONLY_PATTERN = re.compile(
    r"^chapter\s+(?:[ivxlcdm]+|\d+|" + CHAPTER_NUMBERS + r")[\s.:]*$", re.IGNORECASE
)
