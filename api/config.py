import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"


def get_llm():
    if USE_OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
            temperature=0.1,
        )

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
        )

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
    )


def get_embeddings():
    # Groq doesn't provide an embeddings API, so we use HuggingFace regardless
    if USE_OLLAMA or os.getenv("GROQ_API_KEY"):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/ragdb"
)