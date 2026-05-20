import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from api.schemas import QueryRequest, QueryResponse, SourceDocument
from api.config import get_llm
from retrieval.retriever import retrieve_basic, retrieve_mmr, retrieve_hybrid_rerank, format_context

load_dotenv()

app = FastAPI(
    title="RAG Portfolio API",
    description="Production RAG system with RAGAS evaluation",
    version="1.0.0",
)

# ── Prompt Template ──────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."
For every claim you make, cite the source like [1], [2] etc.

Context:
{context}

Question: {question}

Answer:
""")


# ── Helper ───────────────────────────────────────────────────
def get_chunks(question: str, strategy: str, k: int, all_chunks=None):
    if strategy == "hybrid":
        if all_chunks is None:
            raise ValueError("all_chunks required for hybrid strategy")
        return retrieve_hybrid_rerank(question, all_chunks, k=k)
    if strategy == "mmr":
        return retrieve_mmr(question, k=k, fetch_k=k * 4)
    return retrieve_basic(question, k=k)


# ── Routes ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "RAG API is live"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()

    try:
        # Load all chunks only when hybrid strategy is selected
        all_chunks = None
        if request.strategy == "hybrid":
            from ingest.loader import load_documents
            from ingest.chunker import chunk_recursive
            docs       = load_documents("data/raw")
            all_chunks = chunk_recursive(docs)

        chunks = get_chunks(request.question, request.strategy, request.k, all_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # 2. Format context
    context = format_context(chunks)

    # 3. Generate answer
    try:
        llm    = get_llm()
        chain  = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context":  context,
            "question": request.question,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # 4. Build response
    latency = (time.time() - start) * 1000
    sources = [
        SourceDocument(
            source=doc.metadata.get("source", "unknown"),
            content=doc.page_content[:300],
        )
        for doc in chunks
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        strategy_used=request.strategy,
        chunks_retrieved=len(chunks),
        latency_ms=round(latency, 2),
    )


@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "llm":       "ollama/qwen2.5" if os.getenv("USE_OLLAMA", "true") == "true" else "openai/gpt-4o-mini",
        "embeddings": "HuggingFace/MiniLM" if os.getenv("USE_OLLAMA", "true") == "true" else "OpenAI/text-embedding-3-small",
    }