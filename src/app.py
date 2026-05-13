"""FastAPI application for the Readiscoverers backend API."""

import os
import json
import uuid

from pydantic import BaseModel
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import vertexai
from google import genai

from .constants import (
    TEMP_DIR,
    PROCESSED_BOOKS_EMBEDDINGS_DIR,
    PROCESSED_BOOKS_METADATA_DIR,
    USE_GCS,
    GCS_BUCKET_NAME,
    GCS_EMBEDDINGS_PREFIX,
    GCS_METADATA_PREFIX,
)
from .gcs_utils import (
    write_pickle_to_gcs,
    write_json_to_gcs,
    read_pickle_with_cache,
    read_json_with_cache,
)
from .get_book_df import get_book_df
from .model import call_model_with_structured_output
from .search import find_best_text_chunks

PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
LOCATION = str(os.environ.get("GOOGLE_CLOUD_LOCATION"))
PROXY_SECRET = os.environ.get("PROXY_SECRET")

# Localhost only — production traffic comes through the Netlify proxy (server-to-server, no Origin header)
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8888",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def verify_proxy_secret(request: Request, call_next):
    if PROXY_SECRET and request.method == "POST":
        if request.headers.get("x-proxy-secret") != PROXY_SECRET:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)


vertexai.init(project=PROJECT_ID, location=LOCATION)
client = genai.Client(
    vertexai=True, project=PROJECT_ID, location=LOCATION
)  # Changed location to match vertexai.init


class ModelRequest(BaseModel):
    """Request model for model response endpoint."""

    # pylint: disable=too-few-public-methods
    user_query: str = None


class SearchRequest(BaseModel):
    """Request model for search response endpoint."""

    # pylint: disable=too-few-public-methods
    query: str = None
    filenames: list[str] = []
    top_k: int = 3
    query_id: str = None
    enhanced_query: bool = None
    keywords: list[str] = []


class BookDataRequest(BaseModel):
    """Request model for book data processing endpoint."""

    # pylint: disable=too-few-public-methods
    url: str = None
    target_chunk_size: int = 1200
    sentence_overlap: int = 3
    small_paragraph_length: int = 300
    small_paragraph_overlap: int = 3


@app.get("/")
async def root(request: Request):
    """Root endpoint to verify API is running."""
    origin = request.headers.get("origin", "No origin header")
    print(f"Root endpoint called from origin: {origin}")
    return {
        "message": "Hey there! You did not provide an endpoint, but looks like the backend API is healthy."
    }


@app.options("/")
async def options_root():
    """Handle CORS preflight for root endpoint."""
    return Response(status_code=200)


@app.post("/v1/book-data")
async def book_data(req: BookDataRequest):
    """Download and process a book from URL into chunks with embeddings."""
    if not client:
        return {"status": "error", "message": "GenAI client is not initialized."}
    if any(
        param < 0
        for param in [
            req.target_chunk_size,
            req.sentence_overlap,
            req.small_paragraph_length,
            req.small_paragraph_overlap,
        ]
    ):
        return {
            "status": "error",
            "message": "All chunking parameters must be positive integers.",
        }

    book_uuid = str(uuid.uuid4())  # Generate UUID for filename during download
    response = get_book_df(
        url=req.url,
        local_filename=book_uuid,
        target_chunk_size=int(req.target_chunk_size),
        sentence_overlap=int(req.sentence_overlap),
        small_paragraph_length=int(req.small_paragraph_length),
        small_paragraph_overlap=int(req.small_paragraph_overlap),
        client=client,
    )
    if response["status"] == "error":
        return response

    df = response["book_data"]
    filename = response["filename"]
    book_title = response.get("book_title", "Unknown Title")
    book_author = response.get("book_author", "Unknown Author")
    was_cached = response.get("cached", False)

    # Only save if not already cached
    if not was_cached:
        if USE_GCS:
            # Save to GCS
            embeddings_blob = f"{GCS_EMBEDDINGS_PREFIX}/{filename}.pkl"
            metadata_blob = f"{GCS_METADATA_PREFIX}/{filename}_metadata.json"

            success = write_pickle_to_gcs(GCS_BUCKET_NAME, embeddings_blob, df)
            if not success:
                return {
                    "status": "error",
                    "message": "Failed to save embeddings to GCS",
                }

            # Save chunking metadata
            metadata = {
                "target_chunk_size": req.target_chunk_size,
                "sentence_overlap": req.sentence_overlap,
                "small_paragraph_length": req.small_paragraph_length,
                "small_paragraph_overlap": req.small_paragraph_overlap,
                "book_title": book_title,
                "book_author": book_author,
            }

            success = write_json_to_gcs(GCS_BUCKET_NAME, metadata_blob, metadata)
            if not success:
                return {"status": "error", "message": "Failed to save metadata to GCS"}

            if os.environ.get("ENV") == "dev":
                print(
                    f"Saved book '{book_title}' to GCS: {embeddings_blob}, {metadata_blob}"
                )
        else:
            # Save to local filesystem
            os.makedirs(PROCESSED_BOOKS_EMBEDDINGS_DIR, exist_ok=True)
            os.makedirs(PROCESSED_BOOKS_METADATA_DIR, exist_ok=True)

            # Save pickle to embeddings directory
            df.to_pickle(f"{PROCESSED_BOOKS_EMBEDDINGS_DIR}/{filename}.pkl")

            # Save chunking metadata to metadata directory
            metadata = {
                "target_chunk_size": req.target_chunk_size,
                "sentence_overlap": req.sentence_overlap,
                "small_paragraph_length": req.small_paragraph_length,
                "small_paragraph_overlap": req.small_paragraph_overlap,
                "book_title": book_title,
                "book_author": book_author,
            }

            with open(
                f"{PROCESSED_BOOKS_METADATA_DIR}/{filename}_metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(metadata, f)

        # Save debug CSV to temp directory (always, regardless of storage backend)
        if os.environ.get("ENV") == "dev":
            print("Saving CSV for debugging purposes.")
            os.makedirs(TEMP_DIR, exist_ok=True)
            df.to_csv(f"{TEMP_DIR}/_DEV_{filename}.csv", index=False)
    elif os.environ.get("ENV") == "dev":
        print(f"Book '{book_title}' was cached, skipped saving.")

    return {
        "status": "success",
        "filename": filename,
        "book_title": book_title,
        "book_author": book_author,
        "cached": was_cached,
        "message": (
            "Book data processed and saved."
            if not was_cached
            else "Book data retrieved from cache."
        ),
    }


@app.options("/v1/book-data")
async def options_book_data():
    """Handle CORS preflight for book data endpoint."""
    return Response(status_code=200)


@app.post("/v1/search-response")
async def search_response(req: SearchRequest):
    """Search for relevant passages in processed book data."""
    # pylint: disable=too-many-return-statements
    if not req.filenames:
        return {"status": "error", "message": "at least one filename must be provided."}
    if req.query is None:
        return {"status": "error", "message": "query must be provided."}
    if not client:
        return {"status": "error", "message": "GenAI client is not initialized."}
    if req.top_k <= 0:
        return {"status": "error", "message": "top_k must be a positive integer."}
    if req.query_id is None:
        return {"status": "error", "message": "query_id must be provided."}
    if req.enhanced_query is None:
        return {"status": "error", "message": "enhanced_query must be provided."}
    if req.keywords is None:
        return {
            "status": "error",
            "message": "keywords or empty list must be provided.",
        }

    # Load and combine dataframes from all filenames
    combined_dfs_as_list = []
    chunking_metadata = None

    # Load dataframe
    for filename in req.filenames:
        if USE_GCS:
            # Load from GCS with Redis caching
            embeddings_blob = f"{GCS_EMBEDDINGS_PREFIX}/{filename}.pkl"
            metadata_blob = f"{GCS_METADATA_PREFIX}/{filename}_metadata.json"

            df = read_pickle_with_cache(GCS_BUCKET_NAME, embeddings_blob)
            if df is None:
                return {
                    "status": "error",
                    "message": f"Dataframe file not found in GCS: {embeddings_blob}",
                }

            chunking_metadata = read_json_with_cache(GCS_BUCKET_NAME, metadata_blob)
            if chunking_metadata is None:
                if os.environ.get("ENV") == "dev":
                    print(f"Chunking metadata file not found in GCS: {metadata_blob}")
        else:
            # Load from local filesystem
            pickle_path = f"{PROCESSED_BOOKS_EMBEDDINGS_DIR}/{filename}.pkl"
            if not os.path.exists(pickle_path):
                return {
                    "status": "error",
                    "message": f"Dataframe file not found: {pickle_path}",
                }
            df = pd.read_pickle(pickle_path)

            # Load chunking metadata
            # THIS DOESNT NEED TO BE PER BOOK
            metadata_path = f"{PROCESSED_BOOKS_METADATA_DIR}/{filename}_metadata.json"

            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    chunking_metadata = json.load(f)
            elif os.environ.get("ENV") == "dev":
                print(f"Chunking metadata file not found: {metadata_path}")

        # Add book identifier and original index before combining
        df["filename"] = filename
        df["book_chunk_index"] = df.index  # Store original index
        df["book_chunk_length"] = len(df)  # Store total chunks in this book

        if chunking_metadata:
            df["book_title"] = chunking_metadata.get("book_title")
            df["book_author"] = chunking_metadata.get("book_author")

        combined_dfs_as_list.append(df)

    # Combine all dataframes
    combined_books_df = pd.concat(combined_dfs_as_list, ignore_index=True)

    if os.environ.get("ENV") == "dev":
        print(
            f"Combined {len(combined_dfs_as_list)} dataframes with "
            f"total {len(combined_books_df)} rows"
        )
        combined_books_df.to_csv(f"{TEMP_DIR}/_DEV_combined_books_df.csv")

    response = find_best_text_chunks(
        query=req.query,
        combined_books_df=combined_books_df,
        client=client,
        top_k=req.top_k,
        query_id=req.query_id,
        enhanced_query=req.enhanced_query,
        chunking_metadata=chunking_metadata,
        keywords=req.keywords,
    )
    if response["status"] == "error":
        return response

    print(response["search_results"])
    return response


@app.options("/v1/search-response")
async def options_search_response():
    """Handle CORS preflight for search response endpoint."""
    return Response(status_code=200)


@app.post("/v1/model-response")
async def model_response(req: ModelRequest):
    """Convert natural language query into optimized book search query."""
    if req.user_query is None:
        return {
            "status": "error",
            "message": "user_query must be provided",
        }

    structured_result = call_model_with_structured_output(
        user_query=req.user_query, client=client
    )

    if not structured_result:
        # Fall back to original query if there's an error
        search_query = req.user_query
        keywords = []
        if os.environ.get("ENV") == "dev":
            print(f"No structured result, using original query: {req.user_query}")
    else:
        # Use the optimized search query
        search_query = structured_result.get("search_query", req.user_query)
        keywords = structured_result.get("keywords", [])
        if os.environ.get("ENV") == "dev":
            print(f"Original query: {req.user_query}")
            print(f"Optimized query: {search_query}")
            print(f"Keywords: {keywords}")

    return {
        "status": "success",
        "search_query": search_query,
        "keywords": keywords,
    }


@app.options("/v1/model-response")
async def options_model_response():
    """Handle CORS preflight for model response endpoint."""
    return Response(status_code=200)
