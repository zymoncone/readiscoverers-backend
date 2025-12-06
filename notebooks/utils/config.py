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

# Define parameter combinations to test
PARAM_COMBOS = [
    # (target_chunk_size, sentence_overlap, small_paragraph_length, small_paragraph_overlap)
    (800, 2, 200, 2),  # Default
    (1000, 2, 200, 2),  # Larger chunks
    (1000, 2, 200, 3),  # Larger chunks with more paragraph overlap
    (1000, 2, 150, 3),  # Larger chunks with smaller paragraph threshold
    (600, 2, 200, 2),  # Smaller chunks
    (600, 2, 200, 3),  # Smaller chunks with more paragraph overlap
    (600, 2, 150, 3),  # Smaller chunks with smaller
    (600, 2, 250, 3),  # Smaller chunks with larger paragraph threshold
    (700, 2, 200, 2),  # Medium chunks
    (700, 2, 200, 3),  # Medium chunks with more paragraph overlap
    (700, 2, 150, 3),  # Medium chunks with smaller paragraph threshold
    (800, 3, 200, 2),  # More sentence overlap
    (800, 1, 200, 2),  # Less sentence overlap
    (800, 2, 150, 2),  # Smaller paragraph threshold
    (800, 2, 250, 2),  # Larger paragraph threshold
    (800, 2, 200, 3),  # More paragraph overlap
    (800, 2, 200, 1),  # Less paragraph overlap
]
