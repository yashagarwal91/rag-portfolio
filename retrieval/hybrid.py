from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from rank_bm25 import BM25Okapi
from api.config import get_embeddings, DB_URL
import numpy as np

COLLECTION_NAME = "rag_documents"

def get_vectorstore():
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def hybrid_search(
    query: str,
    all_chunks: list[Document],
    k: int = 5,
    alpha: float = 0.5,
) -> list[Document]:
    """
    Hybrid search = dense (vector) + sparse (BM25) combined.

    alpha controls the blend:
      alpha=1.0 → pure vector search
      alpha=0.0 → pure BM25 keyword search
      alpha=0.5 → equal blend (recommended default)

    Why hybrid beats vector-only:
    - Vector search finds semantically similar chunks
    - BM25 finds exact keyword matches (names, IDs, acronyms)
    - Combining both covers cases each misses alone
    """
    vs = get_vectorstore()

    # ── Dense retrieval (vector) ──────────────────────────────
    dense_results = vs.similarity_search_with_score(query, k=k * 2)
    # pgvector returns distance (lower = better), normalize to 0-1
    if dense_results:
        dense_scores  = np.array([score for _, score in dense_results])
        dense_min, dense_max = dense_scores.min(), dense_scores.max()
        if dense_max > dense_min:
            dense_scores = 1 - (dense_scores - dense_min) / (dense_max - dense_min)
        else:
            dense_scores = np.ones(len(dense_scores))
    dense_docs = [doc for doc, _ in dense_results]

    # ── Sparse retrieval (BM25) ───────────────────────────────
    tokenized_corpus = [doc.page_content.lower().split() for doc in all_chunks]
    bm25             = BM25Okapi(tokenized_corpus)
    tokenized_query  = query.lower().split()
    bm25_scores_all  = bm25.get_scores(tokenized_query)

    # Get top-k*2 BM25 results
    top_bm25_indices = np.argsort(bm25_scores_all)[::-1][:k * 2]
    bm25_docs        = [all_chunks[i] for i in top_bm25_indices]
    bm25_scores_top  = bm25_scores_all[top_bm25_indices]

    # Normalize BM25 scores to 0-1
    if bm25_scores_top.max() > 0:
        bm25_scores_top = bm25_scores_top / bm25_scores_top.max()

    # ── Combine scores ────────────────────────────────────────
    combined = {}

    for doc, score in zip(dense_docs, dense_scores):
        key = doc.page_content[:100]  # use content as key
        combined[key] = {
            "doc":         doc,
            "dense_score": float(score),
            "bm25_score":  0.0,
        }

    for doc, score in zip(bm25_docs, bm25_scores_top):
        key = doc.page_content[:100]
        if key in combined:
            combined[key]["bm25_score"] = float(score)
        else:
            combined[key] = {
                "doc":         doc,
                "dense_score": 0.0,
                "bm25_score":  float(score),
            }

    # Weighted combination
    for key in combined:
        d = combined[key]["dense_score"]
        b = combined[key]["bm25_score"]
        combined[key]["final_score"] = alpha * d + (1 - alpha) * b

    # Sort by final score and return top-k
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["final_score"],
        reverse=True,
    )

    print(f"\n🔀 Hybrid search → {len(sorted_results)} candidates → returning top {k}")
    return [r["doc"] for r in sorted_results[:k]]