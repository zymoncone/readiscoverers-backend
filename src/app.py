from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from .rag_model import get_rag_model_response

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


class QueryRequest(BaseModel):
    user_query: str = None
    new_rag_corpus_name: Optional[str] = None


@app.get("/")
async def root(request: Request):
    origin = request.headers.get("origin", "No origin header")
    print(f"Root endpoint called from origin: {origin}")
    return {"message": "Hey there! Welcome to a RAG world."}


@app.options("/")
async def options_root():
    """Handle CORS preflight for root endpoint"""
    return Response(status_code=200)


@app.post("/v1/model-response")
async def model_response(req: QueryRequest, request: Request):
    origin = request.headers.get("origin", "No origin header")
    print(f"Model response endpoint called from origin: {origin}")
    print(f"Query: {req.user_query}")

    try:
        response = get_rag_model_response(req.user_query, req.new_rag_corpus_name)
        return {"user_query": req.user_query, "response": response}
    except Exception as e:
        error_msg = str(e)
        print(f"Error in model_response endpoint: {error_msg}")
        return {
            "user_query": req.user_query,
            "response": f"Error: {error_msg}",
            "error": True,
        }


@app.options("/v1/model-response")
async def options_model_response():
    """Handle CORS preflight for model response endpoint"""
    return Response(status_code=200)
