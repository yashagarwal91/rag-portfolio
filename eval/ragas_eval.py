import os
import json
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from api.config import get_llm
from retrieval.retriever import retrieve_basic, retrieve_mmr, format_context

load_dotenv()


def get_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    )


def get_ragas_embeddings():
    # Groq doesn't provide an embeddings API
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


def run_rag_pipeline(question: str, strategy: str = "mmr", k: int = 5):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    if strategy == "mmr":
        chunks = retrieve_mmr(question, k=k, fetch_k=k * 4)
    else:
        chunks = retrieve_basic(question, k=k)

    context = format_context(chunks)

    prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."
Cite sources like [1], [2] for every claim.

Context:
{context}

Question: {question}

Answer:
""")
    chain  = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return answer, [doc.page_content for doc in chunks]


def evaluate_pipeline(
    testset_path: str = "eval/testset.json",
    strategy: str = "mmr",
    k: int = 5,
):
    print(f"\nEvaluating strategy='{strategy}', k={k}")
    print("-" * 50)

    with open(testset_path) as f:
        testset = json.load(f)

    questions, answers, contexts, references = [], [], [], []

    for i, item in enumerate(testset):
        q = item["user_input"]
        print(f"  [{i+1}/{len(testset)}] {q[:60]}...")

        answer, ctx = run_rag_pipeline(q, strategy=strategy, k=k)

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx)
        references.append(item.get("reference", ""))

        time.sleep(2)

    dataset = Dataset.from_dict({
        "user_input":         questions,
        "response":           answers,
        "retrieved_contexts": contexts,
        "reference":          references,
    })

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
    )

    return result


def _extract_score(val):
    if isinstance(val, list):
        valid = [v for v in val if v == v]
        return sum(valid) / len(valid) if valid else 0.0
    return float(val) if val == val else 0.0


def compare_strategies(testset_path: str = "eval/testset.json"):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    results = {}

    for strategy in ["basic", "mmr"]:
        result   = evaluate_pipeline(testset_path, strategy=strategy)
        cleaned  = {m: _extract_score(result[m]) for m in metrics}
        results[strategy] = cleaned

        with open("eval/results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(f"{'Metric':<25} {'Basic':>15} {'MMR':>15}")
    print("=" * 65)

    for metric in metrics:
        basic = results["basic"][metric]
        mmr   = results["mmr"][metric]
        winner = "<<" if mmr >= basic else ""
        print(f"{metric:<25} {basic:>15.4f} {mmr:>15.4f}  {winner}")

    print("=" * 65)
    os._exit(0)


if __name__ == "__main__":
    compare_strategies()