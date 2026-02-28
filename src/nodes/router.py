from __future__ import annotations

from typing import Literal
import logging

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from ..llm_config import ROUTER_LLM
from ..prompt_loader import load_prompt
from ..state import AgentState


RETRIEVAL_INTENTS = {"busqueda", "resumen", "comparacion"}
VALID_INTENTS = RETRIEVAL_INTENTS | {"general"}
INTENT_ALIASES = {
    "consulta_general": "general",
    "consulta general": "general",
    "search": "busqueda",
    "summary": "resumen",
    "compare": "comparacion",
}

load_dotenv()
logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    intent: Literal["busqueda", "resumen", "comparacion", "general"] = Field(
        description=(
            "Intent de la consulta del usuario: busqueda, resumen, comparacion o general."
        )
    )


def _router_llm() -> ChatGroq:
    return ChatGroq(model=ROUTER_LLM.model, temperature=ROUTER_LLM.temperature)


def _is_memory_update(normalized: str) -> bool:
    has_memory_intent = any(
        token in normalized
        for token in ("recuerda", "recordar", "guarda", "guardar", "almacena", "memoriza", "ten en cuenta")
    )
    if not has_memory_intent:
        if ("mi papa" in normalized or "mi promedio" in normalized) and any(ch.isdigit() for ch in normalized):
            return True
        if "mi semestre" in normalized and any(ch.isdigit() for ch in normalized):
            return True
        if "mi programa" in normalized and "es" in normalized:
            return True
        return False
    if "papa" in normalized or "promedio" in normalized:
        return "es" in normalized or ":" in normalized or "=" in normalized or "de" in normalized
    if "creditos aprobados" in normalized or "semestre actual" in normalized:
        return True
    if "programa es" in normalized:
        return True
    return False

def _heuristic_intent(question: str) -> str | None:
    normalized = question.strip().lower()
    if not normalized:
        return None

    if _is_memory_update(normalized):
        return "general"

    if any(token in normalized for token in ("comparar", "comparación", "comparacion", "contrastar", "diferencia", "vs", "versus")):
        return "comparacion"

    if any(token in normalized for token in ("resumen", "resume", "resumir", "sintetiza", "sintesis", "sintetizar")):
        return "resumen"

    retrieval_terms = (
        "articulo",
        "artículo",
        "norma",
        "acuerdo",
        "resolucion",
        "resolución",
        "estatuto",
        "requisito",
        "requisitos",
        "doble titulacion",
        "titulacion",
        "créditos",
        "creditos",
        "matricula",
        "matrícula",
        "admisión",
        "admision",
        "plazo",
        "semana",
        "procedimiento",
        "proceso",
    )
    if any(token in normalized for token in retrieval_terms) or "?" in normalized:
        return "busqueda"

    return None


def _normalize_intent(raw_intent: str) -> str:
    normalized = raw_intent.strip().lower().replace("-", "_")
    normalized = INTENT_ALIASES.get(normalized, normalized)
    if normalized in VALID_INTENTS:
        return normalized
    return "general"


def classify_intent(state: AgentState) -> AgentState:
    """Classify the user question and store normalized intent in state."""
    question = state.get("question", "").strip()
    if not question:
        return {**state, "intent": "general"}

    if _is_memory_update(question.lower()):
        return {**state, "intent": "general"}

    llm = _router_llm().with_structured_output(IntentClassification)
    prompt = load_prompt("router").format(question=question)
    try:
        result = llm.invoke(prompt)
        normalized = _normalize_intent(result.intent)
    except Exception as exc:
        logger.warning(
            "LLM router failed (possible rate limit or connection issue). "
            "provider=%s model=%s. Falling back to heuristic.",
            ROUTER_LLM.provider,
            ROUTER_LLM.model,
            exc_info=exc,
        )
        normalized = "general"
    if normalized == "general":
        heuristic = _heuristic_intent(question)
        if heuristic:
            normalized = heuristic
    return {**state, "intent": normalized}


def route_by_intent(state: AgentState) -> Literal["k_selector", "direct_llm"]:
    """Route retrieval intents to k_selector, otherwise answer directly."""
    intent = _normalize_intent(str(state.get("intent", "")))
    question = str(state.get("question", "")).strip().lower()
    if _is_memory_update(question):
        return "direct_llm"
    if state.get("memory_updated"):
        return "direct_llm"
    if intent in RETRIEVAL_INTENTS:
        return "k_selector"
    if _heuristic_intent(question) in RETRIEVAL_INTENTS:
        return "k_selector"
    return "direct_llm"
