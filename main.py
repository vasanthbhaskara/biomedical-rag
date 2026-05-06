# ============================================================
# main.py — FastAPI app
# Endpoints:
#   GET  /health       → liveness check
#   POST /query        → RAG query
#   GET  /index-stats  → index info
# ============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

import rag_engine
from rag_engine import init_engine, rag_query


# ── Lifespan: build index once at startup ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up — building RAG index...")
    init_engine(
        query="guselkumab Tremfya psoriasis clinical trial",
        max_results=60,
    )
    print("✅ RAG engine ready")
    yield
    print("👋 Shutting down")


app = FastAPI(
    title="Pharma RAG API",
    description=(
        "Retrieval-Augmented Generation over PubMed abstracts "
        "for guselkumab / Tremfya immunology literature."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        example="What is the mechanism of action of guselkumab?",
    )
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class Source(BaseModel):
    pmid:  str
    title: str
    year:  str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[Source]


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/index-stats", tags=["Meta"])
def index_stats():
    """Returns info about the current retrieval index."""
    if rag_engine.faiss_index is None:
        raise HTTPException(status_code=503, detail="Index not yet built")
    return {
        "total_vectors": rag_engine.faiss_index.ntotal,
        "total_chunks":  len(rag_engine.chunks),
        "embedding_dim": rag_engine.faiss_index.d,
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """
    Main RAG endpoint.
    Retrieves relevant PubMed chunks and generates a grounded answer.
    """
    if rag_engine.faiss_index is None:
        raise HTTPException(status_code=503, detail="Index not yet built")

    try:
        result = rag_query(request.question, k=request.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
    )


# ── Dev runner ────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
