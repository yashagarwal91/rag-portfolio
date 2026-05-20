from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from api.config import get_embeddings


def chunk_recursive(docs, chunk_size=512, overlap=64):
    # Splits on paragraphs -> sentences -> words in order.
    # Respects document structure better than fixed-size splitting.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"recursive  -> {len(chunks)} chunks")
    return chunks


def chunk_fixed(docs, chunk_size=512, overlap=64):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"fixed      -> {len(chunks)} chunks")
    return chunks


def chunk_semantic(docs):
    # Uses embeddings to detect topic shifts and split at natural boundaries.
    # Slower than the other two but produces more coherent chunks for
    # dense technical or legal documents.
    splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chunks = splitter.split_documents(docs)
    print(f"semantic   -> {len(chunks)} chunks")
    return chunks


def compare_strategies(docs):
    print("\nChunking Strategy Comparison")
    print("-" * 55)
    print(f"{'Strategy':<12} {'Chunks':>8} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)

    results = {}
    for name, chunks in [
        ("recursive", chunk_recursive(docs)),
        ("fixed",     chunk_fixed(docs)),
        ("semantic",  chunk_semantic(docs)),
    ]:
        sizes = [len(c.page_content) for c in chunks]
        avg   = sum(sizes) // len(sizes)
        print(f"{name:<12} {len(chunks):>8} {avg:>8} {min(sizes):>8} {max(sizes):>8}")
        results[name] = chunks

    print("-" * 55)
    return results


if __name__ == "__main__":
    import os
    from ingest.loader import load_documents

    docs    = load_documents("data/raw")
    results = compare_strategies(docs)

    sample = results["recursive"][0]
    print(f"\nsource  : {sample.metadata.get('source', 'unknown')}")
    print(f"length  : {len(sample.page_content)} chars")
    print(f"preview : {sample.page_content[:300]}...")

    os._exit(0)