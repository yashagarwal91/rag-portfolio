import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)


def load_documents(data_dir: str = "data/raw") -> list:
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")

    documents = []

    loaders = [
        DirectoryLoader(data_dir, glob="**/*.pdf",  loader_cls=PyPDFLoader,           show_progress=True),
        DirectoryLoader(data_dir, glob="**/*.txt",  loader_cls=TextLoader,            show_progress=True),
        DirectoryLoader(data_dir, glob="**/*.html", loader_cls=UnstructuredHTMLLoader, show_progress=True),
    ]

    for loader in loaders:
        documents.extend(loader.load())

    print(f"loaded {len(documents)} documents from '{data_dir}'")
    return documents


if __name__ == "__main__":
    docs = load_documents("data/raw")
    for i, doc in enumerate(docs[:3]):
        print(f"\n[{i+1}] {doc.metadata.get('source', 'unknown')}")
        print(f"    length  : {len(doc.page_content)} chars")
        print(f"    preview : {doc.page_content[:200]}...")

    # os._exit avoids a Windows threading bug on shutdown (Python 3.12 + pypdf)
    os._exit(0)