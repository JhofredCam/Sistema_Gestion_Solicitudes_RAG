from __future__ import annotations

from langchain_core.tools import tool
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

from ..llm_config import RAG_GENERATION_LLM
from ..unal_rag.utils.errors import is_rate_limit_429
from ..prompt_loader import load_prompt


@tool
def resumir_norma(contexto: str, pregunta: str) -> str:
    """Genera un resumen usando el contexto recuperado."""
    logger = logging.getLogger(__name__)
    llm = ChatGoogleGenerativeAI(
        model=RAG_GENERATION_LLM.model,
        temperature=RAG_GENERATION_LLM.temperature,
        # google-genai treats 0 as falsy and falls back to default retries
        max_retries=1,
    )
    prompt = load_prompt("rag_summary").format(question=pregunta, context=contexto)
    try:
        response = llm.invoke(prompt)
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        logger.warning(
            "Summary LLM failed (possible rate limit or connection issue). "
            "provider=%s model=%s.",
            RAG_GENERATION_LLM.provider,
            RAG_GENERATION_LLM.model,
            exc_info=exc,
        )
        if is_rate_limit_429(exc):
            return "No fue posible generar el resumen por limite de tasa (429)."
        return "No fue posible generar el resumen en este momento."
