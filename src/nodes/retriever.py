from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

from ..llm_config import K_SELECTOR_LLM
from ..state import AgentState


PERSIST_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_K = 4
MIN_K = 2
MAX_K = 8
DEFAULT_MAX_ITERATIONS = 2

load_dotenv()


class KSelection(BaseModel):
    k_value: int = Field(
        ge=MIN_K,
        le=MAX_K,
        description="Cantidad de chunks a recuperar de la base vectorial.",
    )


def _build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)


def _k_selector_llm() -> ChatGroq:
    return ChatGroq(model=K_SELECTOR_LLM.model, temperature=K_SELECTOR_LLM.temperature)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_k(k_value: int) -> int:
    return max(MIN_K, min(MAX_K, k_value))


def _truncate(text: str, max_chars: int = 260) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def select_k_node(state: AgentState) -> AgentState:
    """Choose retrieval k dynamically based on intent and question complexity."""
    question = state.get("question", "").strip()
    intent = str(state.get("intent", "busqueda")).strip().lower()

    fallback_by_intent = {
        "comparacion": 6,
        "resumen": 5,
        "busqueda": 4,
    }
    fallback_k = fallback_by_intent.get(intent, DEFAULT_K)

    if not question:
        selected_k = fallback_k
    else:
        prompt = (
            "Eres un selector de k para retrieval semantico en normativa universitaria.\\n"
            "Devuelve solo un entero k entre 2 y 8.\\n"
            "Reglas: comparacion suele requerir mas contexto; busqueda puntual menos.\\n"
            f"Intent: {intent}\\n"
            f"Consulta: {question}"
        )
        try:
            result = _k_selector_llm().with_structured_output(KSelection).invoke(prompt)
            selected_k = _clamp_k(result.k_value)
        except Exception:
            selected_k = fallback_k

    iteration_count = _safe_int(state.get("iteration_count", 0), 0)
    max_iterations = _safe_int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS), DEFAULT_MAX_ITERATIONS)

    return {
        **state,
        "k_value": selected_k,
        "iteration_count": max(0, iteration_count),
        "max_iterations": max(0, max_iterations),
    }


def retriever_node(state: AgentState) -> AgentState:
    """Retrieve relevant documents from vector DB for RAG intents."""
    question = state.get("question", "").strip()
    if not question:
        return {**state, "documents": [], "sources": [], "retrieval_trace": []}

    k_value = _clamp_k(_safe_int(state.get("k_value", DEFAULT_K), DEFAULT_K))

    try:
        vectorstore = _build_vectorstore()
        documents = vectorstore.similarity_search(question, k=k_value)
    except Exception:
        documents = []

    raw_sources = [str(doc.metadata.get("source", "unknown_source")) for doc in documents]
    sources = list(dict.fromkeys(raw_sources))

    retrieval_trace: list[dict[str, Any]] = []
    for rank, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        retrieval_trace.append(
            {
                "rank": rank,
                "score": metadata.get("score"),
                "source": str(metadata.get("source", "unknown_source")),
                "doc_id": str(metadata.get("doc_id", metadata.get("source", "unknown_doc"))),
                "chunk_id": str(metadata.get("chunk_id", "unknown_chunk")),
                "page": metadata.get("page", metadata.get("pagina")),
                "snippet": _truncate(doc.page_content),
            }
        )

    return {
        **state,
        "documents": documents,
        "sources": sources,
        "k_value": k_value,
        "retrieval_trace": retrieval_trace,
    }
