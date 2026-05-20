from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Downloads ~67MB on first run, cached after that
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("loading reranker model...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    docs: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """
    Cross-encoder reranking — encodes query+doc together for more accurate
    relevance scoring than bi-encoder retrieval. Slower but worth it as a
    second-stage filter over a larger candidate set.
    """
    if not docs:
        return docs

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:top_n]]