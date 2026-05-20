import os
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from api.config import get_embeddings, DB_URL
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank

COLLECTION_NAME = "rag_documents"


def get_vectorstore():
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def retrieve_basic(query: str, k: int = 5) -> list[Document]:
    return get_vectorstore().similarity_search(query, k=k)


def retrieve_mmr(query: str, k: int = 5, fetch_k: int = 20) -> list[Document]:
    # Fetches fetch_k candidates then picks k most diverse to reduce redundancy
    return get_vectorstore().max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def retrieve_with_score(query: str, k: int = 5) -> list[tuple[Document, float]]:
    return get_vectorstore().similarity_search_with_score(query, k=k)


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i+1}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve_hybrid_rerank(
    query: str,
    all_chunks: list[Document],
    k: int = 5,
) -> list[Document]:
    # Two-stage: broad hybrid retrieval -> precise cross-encoder reranking
    candidates = hybrid_search(query, all_chunks, k=20, alpha=0.5)
    return rerank(query, candidates, top_n=k)


if __name__ == "__main__":
    query = "what is this document about"

    print(f"\nquery: '{query}'")

    print("\n-- basic --")
    basic_results = retrieve_basic(query, k=5)
    for i, doc in enumerate(basic_results):
        print(f"[{i+1}] {doc.metadata.get('source', '?')} | {len(doc.page_content)} chars")
        print(f"     {doc.page_content[:150]}...")

    print("\n-- mmr --")
    for i, doc in enumerate(retrieve_mmr(query, k=5, fetch_k=20)):
        print(f"[{i+1}] {doc.metadata.get('source', '?')} | {len(doc.page_content)} chars")
        print(f"     {doc.page_content[:150]}...")

    print("\n-- scores --")
    for doc, score in retrieve_with_score(query, k=5):
        print(f"{score:.4f} | {doc.page_content[:100]}...")

    print("\n-- formatted context --")
    print(format_context(basic_results[:3]))

    os._exit(0)