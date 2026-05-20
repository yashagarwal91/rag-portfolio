from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    question: str
    strategy: str = "mmr"        # "basic" or "mmr"
    k: int = 5                   # number of chunks to retrieve


class SourceDocument(BaseModel):
    source: str
    content: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    strategy_used: str
    chunks_retrieved: int
    latency_ms: float