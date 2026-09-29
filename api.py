"""FastAPI service. Run: uvicorn api:app --reload  (docs at /docs)"""
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_graph import RAGPipeline

app = FastAPI(title="Agentic AI eBook RAG Chatbot", version="1.0.0")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["What is Agentic AI?"])


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        return get_pipeline().run(req.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
