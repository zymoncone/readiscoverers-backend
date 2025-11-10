from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import vertexai
from google import genai
import os

from .constants import PROJECT_ID, TEMP_DIR
from .get_book_df import get_book_df
from .rag_model import get_rag_model_response
from .search import find_best_passage

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


vertexai.init(project=PROJECT_ID, location="us-east4")
client = genai.Client(
    vertexai=True, project=PROJECT_ID, location="us-east4"
)  # Changed location to match vertexai.init


class QueryRequest(BaseModel):
    user_query: str = None
    new_rag_corpus_name: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = None
    local_filename: str = None
    top_k: int = 3


class BookDataRequest(BaseModel):
    url: str = None
    local_filename: str = None
    chunk_size: int = 1200


@app.get("/")
async def root(request: Request):
    origin = request.headers.get("origin", "No origin header")
    print(f"Root endpoint called from origin: {origin}")
    return {"message": "Hey there! Welcome to a RAG world."}


@app.options("/")
async def options_root():
    """Handle CORS preflight for root endpoint"""
    return Response(status_code=200)


@app.post("/v1/book-data")
async def book_data(req: BookDataRequest):
    response = get_book_df(
        url=req.url,
        local_filename=req.local_filename,
        chunk_size=req.chunk_size,
        client=client,
    )
    if response["status"] == "error":
        return response
    df = response["data"]
    # Save to /tmp for Docker compatibility
    os.makedirs(TEMP_DIR, exist_ok=True)
    df.to_pickle(f"{TEMP_DIR}/{req.local_filename}.pkl")
    return {"status": "success", "message": "Book data processed and saved."}


@app.options("/v1/book-data")
async def options_book_data():
    """Handle CORS preflight for book data endpoint"""
    return Response(status_code=200)


@app.post("/v1/search-response")
async def search_response(req: SearchRequest):
    if req.local_filename is None:
        return {"status": "error", "message": "local_filename must be provided."}
    if req.query is None:
        return {"status": "error", "message": "query must be provided."}
    if not client:
        return {"status": "error", "message": "GenAI client is not initialized."}
    # Read from /tmp for Docker compatibility
    df = pd.read_pickle(f"{TEMP_DIR}/{req.local_filename}.pkl")
    return find_best_passage(
        query=req.query, dataframe=df, client=client, top_k=req.top_k
    )


@app.options("/v1/search-response")
async def options_search_response():
    """Handle CORS preflight for search response endpoint"""
    return Response(status_code=200)


@app.post("/v1/model-response")
async def model_response(req: QueryRequest):
    return {"status": "success", "message": "temporary placeholder response."}


@app.options("/v1/model-response")
async def options_model_response():
    """Handle CORS preflight for model response endpoint"""
    return Response(status_code=200)
