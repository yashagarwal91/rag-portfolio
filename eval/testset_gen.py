import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from api.config import get_llm, get_embeddings
from ingest.loader import load_documents

load_dotenv()


def generate_testset(
    docs: list[Document],
    test_size: int = 20,
    output_path: str = "eval/testset.json"
):
    # Pass full documents, not chunks — RAGAS needs sufficient context per doc
    llm        = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    testset   = generator.generate_with_langchain_docs(docs, testset_size=test_size)

    df = testset.to_pandas()
    print(f"Generated {len(df)} test questions")
    print(df[["user_input", "reference"]].head(5))

    records = df[["user_input", "reference", "reference_contexts"]].to_dict("records")
    os.makedirs("eval", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Testset saved to {output_path}")
    return records


if __name__ == "__main__":
    docs = load_documents("data/raw")
    generate_testset(docs, test_size=10)
    os._exit(0)