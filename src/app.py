"""FastAPI application for the Readiscoverers backend API."""

import os
import json

from pydantic import BaseModel
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import vertexai
from google import genai

from .constants import TEMP_DIR
from .get_book_df import get_book_df
from .model import call_model_with_structured_output
from .search import find_best_text_chunks

PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
LOCATION = str(os.environ.get("GOOGLE_CLOUD_LOCATION"))

# All origins (production + localhost for testing)
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://192.168.50.143:3000",
    "https://readiscover.app",
    "https://www.readiscover.app",
]

app = FastAPI()

# Add CORS middleware - this will work when accessing Cloud Run directly
# and will also add headers that work through API Gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


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
    local_filename: str = None
    top_k: int = 3
    query_id: str = None
    enhanced_query: bool = None


class BookDataRequest(BaseModel):
    """Request model for book data processing endpoint."""

    # pylint: disable=too-few-public-methods
    url: str = None
    local_filename: str = None
    target_chunk_size: int = 800
    sentence_overlap: int = 2
    small_paragraph_length: int = 200
    small_paragraph_overlap: int = 2


@app.get("/")
async def root(request: Request):
    """Root endpoint to verify API is running."""
    origin = request.headers.get("origin", "No origin header")
    print(f"Root endpoint called from origin: {origin}")
    return {"message": "Hey there! Looks like the backend API is healthy."}


@app.options("/")
async def options_root():
    """Handle CORS preflight for root endpoint."""
    return Response(status_code=200)


@app.post("/v1/book-data")
async def book_data(req: BookDataRequest):
    """Download and process a book from URL into chunks with embeddings."""
    if req.local_filename is None:
        return {"status": "error", "message": "local_filename must be provided."}
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

    response = get_book_df(
        url=req.url,
        local_filename=req.local_filename,
        target_chunk_size=int(req.target_chunk_size),
        sentence_overlap=int(req.sentence_overlap),
        small_paragraph_length=int(req.small_paragraph_length),
        small_paragraph_overlap=int(req.small_paragraph_overlap),
        client=client,
    )
    if response["status"] == "error":
        return response

    df = response["book_data"]

    if book_title := response.get("book_title"):
        filename = book_title.replace(" ", "_").lower()
    else:
        filename = req.local_filename

    # Save both the dataframe AND metadata
    os.makedirs(TEMP_DIR, exist_ok=True)
    df.to_pickle(f"{TEMP_DIR}/{filename}.pkl")

    if os.environ.get("ENV") == "dev":
        print("Saving CSV for debugging purposes.")
        df.to_csv(f"{TEMP_DIR}/{filename}.csv", index=False)
    # Save chunking metadata separately
    metadata = {
        "target_chunk_size": req.target_chunk_size,
        "sentence_overlap": req.sentence_overlap,
        "small_paragraph_length": req.small_paragraph_length,
        "small_paragraph_overlap": req.small_paragraph_overlap,
        "book_title": response["book_title"],
        "book_author": response["book_author"],
    }

    with open(f"{TEMP_DIR}/{filename}_metadata.json", "w") as f:
        json.dump(metadata, f)

    return {
        "status": "success",
        "filename": filename,
        "book_title": response["book_title"],
        "book_author": response["book_author"],
        "message": "Book data processed and saved.",
    }


@app.options("/v1/book-data")
async def options_book_data():
    """Handle CORS preflight for book data endpoint."""
    return Response(status_code=200)


@app.post("/v1/search-response")
async def search_response(req: SearchRequest):
    """Search for relevant passages in processed book data."""
    if req.local_filename is None:
        return {"status": "error", "message": "local_filename must be provided."}
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

    # Load dataframe
    pickle_path = f"{TEMP_DIR}/{req.local_filename}.pkl"
    if not os.path.exists(pickle_path):
        return {
            "status": "error",
            "message": f"Dataframe file not found: {pickle_path}",
        }
    df = pd.read_pickle(pickle_path)

    # Load chunking metadata
    metadata_path = f"{TEMP_DIR}/{req.local_filename}_metadata.json"
    chunking_metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            chunking_metadata = json.load(f)
    elif os.environ.get("ENV") == "dev":
        print(f"Chunking metadata file not found: {metadata_path}")

    search_results = find_best_text_chunks(
        query=req.query,
        dataframe=df,
        client=client,
        top_k=req.top_k,
        query_id=req.query_id,
        enhanced_query=req.enhanced_query,
        chunking_metadata=chunking_metadata,
    )
    print(search_results)

    return {"status": "success", "search_results": search_results}


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
