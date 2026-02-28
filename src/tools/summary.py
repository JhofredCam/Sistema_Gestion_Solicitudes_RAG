from __future__ import annotations

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from ..llm_config import RAG_GENERATION_LLM
from ..prompt_loader import load_prompt


@tool
def resumir_norma(contexto: str, pregunta: str) -> str:
    """Genera un resumen usando el contexto recuperado."""
    llm = ChatGoogleGenerativeAI(
        model=RAG_GENERATION_LLM.model,
        temperature=RAG_GENERATION_LLM.temperature,
    )
    prompt = load_prompt("rag_summary").format(question=pregunta, context=contexto)
    response = llm.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)
