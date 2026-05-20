from langchain_postgres.vectorstores import PGVector
from api.config import get_embeddings, DB_URL


COLLECTION_NAME = "rag_documents"


def get_vectorstore():
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
    )


def embed_and_store(chunks):
    # pre_delete_collection ensures no duplicate chunks on re-runs
    vectorstore = PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    batch_size = 50
    total_batches = -(-len(chunks) // batch_size)
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
        print(f"batch {i//batch_size + 1}/{total_batches} stored")

    print(f"{len(chunks)} chunks embedded and stored")
    return vectorstore


if __name__ == "__main__":
    import os
    from ingest.loader import load_documents
    from ingest.chunker import chunk_recursive

    docs   = load_documents("data/raw")
    chunks = chunk_recursive(docs)
    vs     = embed_and_store(chunks)

    query   = "what are mentioned skills"
    results = vs.similarity_search(query, k=3)

    print(f"\ntop {len(results)} results for: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] {doc.metadata.get('source', 'unknown')}")
        print(f"    {doc.page_content[:200]}...")

    os._exit(0)