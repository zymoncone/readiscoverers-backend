from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
from .rag_model import get_rag_model_response

app = FastAPI()


class QueryRequest(BaseModel):
    user_query: str = None
    new_rag_corpus_name: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Hey there! Welcome to a RAG world."}


@app.post("/v1/model-response")
async def model_response(req: QueryRequest):
    response = get_rag_model_response(req.user_query, req.new_rag_corpus_name)
    return {"user_query": req.user_query, "response": response}
