"""Configuration settings for the notebook utilities."""

# Base URL for the local server
BASE_URL = "http://localhost:8080"

# List of book URLs
BOOK_URLS = [
    "https://www.gutenberg.org/cache/epub/55/pg55-images.html",  # The Wonderful Wizard of Oz
    "https://www.gutenberg.org/cache/epub/54/pg54-images.html",  # Marvelous Land of Oz
    "https://www.gutenberg.org/cache/epub/33361/pg33361-images.html",  # Ozma of Oz
    "https://www.gutenberg.org/cache/epub/22566/pg22566-images.html",  # Dorothy and the Wizard in Oz
    "https://www.gutenberg.org/cache/epub/26624/pg26624-images.html",  # The Road to Oz
    "https://www.gutenberg.org/cache/epub/41667/pg41667-images.html",  # The Emerald City of Oz
    "https://www.gutenberg.org/cache/epub/32094/pg32094-images.html",  # The Patchwork Girl of Oz
    "https://www.gutenberg.org/cache/epub/75720/pg75720-images.html",  # Jack Pumpkinhead of Oz
]

TEST_PARAM_COMBOS = [
    # (target_chunk_size, sentence_overlap, small_paragraph_length, small_paragraph_overlap)
    (1200, 3, 300, 3)
]
# Define parameter combinations to test
DEV_PARAM_COMBOS = [
    # (target_chunk_size, sentence_overlap, small_paragraph_length, small_paragraph_overlap)
    (1200, 4, 300, 4),
    (1200, 5, 300, 5),
    (1200, 3, 300, 5),
    (800, 2, 200, 2),  # Default
    (1000, 3, 400, 3),
    (1000, 4, 400, 4),
    (1000, 4, 300, 4),
    (1200, 3, 400, 3),
    (1100, 3, 400, 3),
    (1100, 3, 300, 3),
    (1000, 3, 300, 3),
    (1000, 4, 300, 3),
    (1200, 2, 200, 2),
    (1200, 3, 300, 3),
    (1200, 2, 200, 2),
    (800, 0, 200, 0),
    (800, 0, 200, 4),
    (800, 3, 200, 2),
    (800, 4, 300, 3),
    (800, 3, 150, 3),
    (800, 2, 250, 2),
    (1000, 2, 200, 2),
    (1000, 2, 200, 3),
    (1000, 2, 150, 3),
    (600, 2, 200, 2),
    (600, 2, 200, 3),
    (600, 2, 150, 3),
    (600, 2, 250, 3),
    (700, 2, 200, 2),
    (700, 2, 200, 3),
    (700, 2, 150, 3),
    (800, 2, 200, 3),
    (800, 2, 200, 1),
]
