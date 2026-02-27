from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from ..llm_config import ROUTER_LLM
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


class IntentClassification(BaseModel):
    intent: Literal["busqueda", "resumen", "comparacion", "general"] = Field(
        description=(
            "Intent de la consulta del usuario: busqueda, resumen, comparacion o general."
        )
    )


def _router_llm() -> ChatGroq:
    return ChatGroq(model=ROUTER_LLM.model, temperature=ROUTER_LLM.temperature)


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

    llm = _router_llm().with_structured_output(IntentClassification)
    prompt = (
        "Clasifica la consulta del usuario en una sola etiqueta.\\n"
        "Etiquetas posibles: busqueda, resumen, comparacion, general.\\n"
        "- busqueda: localizar una norma, articulo o requisito especifico.\\n"
        "- resumen: sintetizar contenido normativo de uno o varios documentos.\\n"
        "- comparacion: contrastar reglas o condiciones entre normas.\\n"
        "- general: conversacion o consulta que no requiere retrieval.\\n\\n"
        f"Consulta: {question}"
    )
    result = llm.invoke(prompt)
    return {**state, "intent": _normalize_intent(result.intent)}


def route_by_intent(state: AgentState) -> Literal["k_selector", "direct_llm"]:
    """Route retrieval intents to k_selector, otherwise answer directly."""
    intent = _normalize_intent(str(state.get("intent", "")))
    if intent in RETRIEVAL_INTENTS:
        return "k_selector"
    return "direct_llm"
