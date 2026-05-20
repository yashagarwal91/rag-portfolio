import json
import time

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .source-card {
        background: #181825;
        border-left: 3px solid #89b4fa;
        border-radius: 4px;
        padding: 12px;
        margin: 8px 0;
        font-size: 13px;
    }
    .latency-tag {
        background: #313244;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        color: #cdd6f4;
    }
</style>
""", unsafe_allow_html=True)


def load_eval_results():
    try:
        with open("eval/results.json") as f:
            return json.load(f)
    except Exception:
        return None


def query_api(question: str, strategy: str, k: int):
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "strategy": strategy, "k": k},
            timeout=120,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


with st.sidebar:
    st.title("Configuration")
    st.divider()

    strategy = st.selectbox(
        "Retrieval Strategy",
        ["hybrid", "mmr", "basic"],
        help="Hybrid = BM25 + Vector + Reranking (best quality). MMR = diverse results. Basic = fastest."
    )

    k = st.slider(
        "Chunks to Retrieve (k)",
        min_value=2, max_value=10, value=5,
        help="More chunks = more context but slower"
    )

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("LLM: Groq llama-3.3-70b")
    st.markdown("Embeddings: MiniLM-L6-v2")
    st.markdown("Vector DB: pgvector")
    st.markdown("Reranking: Cross-encoder")

    st.divider()
    try:
        requests.get(f"{API_URL}/health", timeout=3)
        st.success("API: Online")
    except Exception:
        st.error("API: Offline — run uvicorn first")

    st.divider()
    st.markdown("**Built by:** Yash")
    st.markdown("**Stack:** LangChain · FastAPI · pgvector · RAGAS")


st.title("Production RAG — Document QA System")
st.markdown("*Hybrid search · Cross-encoder reranking · Citation-based answers · RAGAS evaluated*")
st.divider()

tab1, tab2, tab3 = st.tabs(["Ask a Question", "RAGAS Evaluation", "Architecture"])


with tab1:
    st.markdown("### Ask anything about your documents")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What are the key guidelines mentioned in the document?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        st.markdown(
            f"<span class='latency-tag'>Strategy: {strategy.upper()} | k={k}</span>",
            unsafe_allow_html=True
        )

    if ask and not question:
        st.warning("Please enter a question first.")

    if ask and question:
        with st.spinner("Retrieving and generating answer..."):
            result = query_api(question, strategy, k)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.markdown("#### Answer")
            st.markdown(
                f"<div style='background:#1e1e2e;padding:16px;border-radius:8px;"
                f"border-left:3px solid #a6e3a1;font-size:15px;line-height:1.7'>"
                f"{result['answer']}</div>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Latency",       f"{result['latency_ms']} ms")
            m2.metric("Chunks Used",   result['chunks_retrieved'])
            m3.metric("Strategy",      result['strategy_used'].upper())

            st.markdown("#### Retrieved Sources")
            for i, src in enumerate(result['sources']):
                with st.expander(f"[{i+1}] {src['source'].split('/')[-1]}"):
                    st.markdown(
                        f"<div class='source-card'>{src['content']}</div>",
                        unsafe_allow_html=True
                    )


with tab2:
    st.markdown("### RAGAS Evaluation Results")
    st.markdown("*Basic similarity vs MMR across 4 metrics*")

    results = load_eval_results()

    if not results:
        st.warning("No eval/results.json found. Run `python -m eval.ragas_eval` first.")
    else:
        metrics = {
            "faithfulness":      "Is the answer grounded in the retrieved context?",
            "answer_relevancy":  "Does the answer actually address the question?",
            "context_precision": "Are retrieved chunks relevant to the question?",
            "context_recall":    "Did we retrieve all necessary information?",
        }

        st.markdown("#### Basic vs MMR")
        for metric, description in metrics.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            basic = results.get("basic", {}).get(metric, 0)
            mmr   = results.get("mmr",   {}).get(metric, 0)

            with col1:
                st.markdown(f"**{metric}**")
                st.caption(description)
            with col2:
                st.metric("Basic", f"{basic:.3f}")
            with col3:
                st.metric("MMR", f"{mmr:.3f}", delta=f"{mmr - basic:+.3f}")

            st.divider()

        st.markdown("#### Interpretation")
        st.info("""
- **Faithfulness = 1.0 (Basic)** — answers are fully grounded in context, no hallucination
- **Answer Relevancy** — MMR scores higher (0.48 vs 0.36), more diverse chunks retrieved
- **Low context scores** — expected with a small corpus; improves significantly with more documents
- **Rate limit impact** — Groq free tier 429s affected MMR eval scores mid-evaluation
        """)

        st.markdown("#### Raw Results")
        st.json(results)


with tab3:
    st.markdown("### System Architecture")
    st.markdown("""
User Query
    |
    v
Streamlit UI  (frontend/app.py)
    |
    | HTTP POST /query
    v
FastAPI App  (api/main.py)
    |
    +------------------+
    v                  v
Retriever          LLM (Groq)
(pgvector)     llama-3.3-70b
    |
    +-- Basic similarity
    +-- MMR
    +-- Hybrid (BM25 + Vector -> Cross-encoder rerank)
    |
    v
pgvector / PostgreSQL  (Docker)
    |
Embeddings: MiniLM-L6-v2
""")

    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Ingestion**
- Loader: LangChain + PyPDF
- Chunking: Recursive / Fixed / Semantic
- Embeddings: all-MiniLM-L6-v2
- Vector store: pgvector (PostgreSQL)

**Retrieval**
- Basic cosine similarity
- MMR (Maximal Marginal Relevance)
- Hybrid: BM25 + dense + cross-encoder rerank
        """)
    with col2:
        st.markdown("""
**Generation**
- LLM: Groq llama-3.3-70b
- Prompt: citation-enforced RAG template
- Output: answer + source references

**Evaluation**
- Framework: RAGAS
- Metrics: faithfulness, answer relevancy,
  context precision, context recall
- Judge: Groq llama-3.3-70b
        """)