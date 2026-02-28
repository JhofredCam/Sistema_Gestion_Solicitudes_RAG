from __future__ import annotations

from typing import Literal
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..llm_config import GROUNDING_EVALUATOR_LLM
from ..prompt_loader import load_prompt
from ..state import AgentState
from .retriever import DEFAULT_K, MAX_K, MIN_K
from ..unal_rag.utils.errors import is_rate_limit_429


DEFAULT_MAX_ITERATIONS = 2
RETRY_K_STEP = 2
MAX_DOC_CHARS = 1200

load_dotenv()
logger = logging.getLogger(__name__)


class GroundingEvaluation(BaseModel):
    is_grounded: bool = Field(
        description="True si la respuesta esta completamente respaldada por el contexto."
    )
    reason: str = Field(description="Justificacion breve de la evaluacion.")
    citation_compliance: bool = Field(
        description="True si cada afirmacion relevante tiene referencias [DOC n] validas."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Afirmaciones no soportadas por el contexto.",
    )


def _evaluator_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GROUNDING_EVALUATOR_LLM.model,
        temperature=GROUNDING_EVALUATOR_LLM.temperature,
        # google-genai treats 0 as falsy and falls back to default retries
        max_retries=1,
    )


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp_k(k_value: int) -> int:
    return max(MIN_K, min(MAX_K, k_value))


def _truncate(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _insufficient_evidence_answer(question: str, reason: str) -> str:
    return (
        "Evidencia insuficiente para responder con certeza usando solo el contexto recuperado.\n"
        f"Consulta: {question}\n"
        f"Motivo: {reason}\n"
        "Sugerencia: reformula la pregunta o solicita documentos mas especificos."
    )


def _append_iteration_history(
    state: AgentState,
    *,
    iteration_count: int,
    k_value: int,
    is_grounded: bool,
    decision: str,
    reason: str,
) -> list[dict]:
    history = list(state.get("iteration_history", []))
    history.append(
        {
            "iteration": iteration_count,
            "k_value": k_value,
            "is_grounded": is_grounded,
            "decision": decision,
            "reason": reason,
        }
    )
    return history


def evaluate_grounding_node(state: AgentState) -> AgentState:
    """Validate answer grounding and decide whether to retry or end."""
    question = state.get("question", "").strip()
    generation = state.get("generation", "").strip()
    documents = state.get("documents", [])
    k_value = _clamp_k(_safe_int(state.get("k_value", DEFAULT_K), DEFAULT_K))

    if state.get("llm_failure"):
        reason = "Fallo de conexion con LLM; no se reintenta."
        iteration_count = max(0, _safe_int(state.get("iteration_count", 0), 0))
        iteration_history = _append_iteration_history(
            state,
            iteration_count=iteration_count,
            k_value=k_value,
            is_grounded=False,
            decision="end",
            reason=reason,
        )
        return {
            **state,
            "is_grounded": False,
            "evaluation_decision": "end",
            "evaluation_result": {"is_grounded": False, "reason": reason},
            "critique_result": {"is_grounded": False, "reason": reason},
            "retry_count": iteration_count,
            "iteration_history": iteration_history,
            "evaluator_prompt": "",
        }

    iteration_count = max(0, _safe_int(state.get("iteration_count", 0), 0))
    max_iterations = max(
        0,
        _safe_int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS), DEFAULT_MAX_ITERATIONS),
    )

    if not question or not generation:
        decision = "end"
        reason = "No hay pregunta o respuesta para verificar."
        iteration_history = _append_iteration_history(
            state,
            iteration_count=iteration_count,
            k_value=k_value,
            is_grounded=False,
            decision=decision,
            reason=reason,
        )
        return {
            **state,
            "is_grounded": False,
            "evaluation_decision": decision,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "generation": _insufficient_evidence_answer(
                question or "consulta vacia",
                reason,
            ),
            "evaluation_result": {"is_grounded": False, "reason": reason},
            "critique_result": {"is_grounded": False, "reason": reason},
            "retry_count": iteration_count,
            "iteration_history": iteration_history,
            "evaluator_prompt": "",
        }

    if not documents:
        if iteration_count < max_iterations:
            next_iteration = iteration_count + 1
            next_k = _clamp_k(k_value + RETRY_K_STEP)
            decision = "retry"
            reason = "Sin documentos recuperados; se reintenta con k mas alto."
            iteration_history = _append_iteration_history(
                state,
                iteration_count=next_iteration,
                k_value=next_k,
                is_grounded=False,
                decision=decision,
                reason=reason,
            )
            return {
                **state,
                "is_grounded": False,
                "evaluation_decision": decision,
                "iteration_count": next_iteration,
                "max_iterations": max_iterations,
                "k_value": next_k,
                "evaluation_result": {"is_grounded": False, "reason": reason},
                "critique_result": {"is_grounded": False, "reason": reason},
                "retry_count": next_iteration,
                "iteration_history": iteration_history,
                "evaluator_prompt": "",
            }

        reason = "No se recupero contexto suficiente para validar la respuesta."
        final_generation = generation
        final_generation = _insufficient_evidence_answer(
            question,
            reason,
        )
        decision = "end"
        iteration_history = _append_iteration_history(
            state,
            iteration_count=iteration_count,
            k_value=k_value,
            is_grounded=False,
            decision=decision,
            reason=reason,
        )
        return {
            **state,
            "is_grounded": False,
            "evaluation_decision": decision,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "generation": final_generation,
            "evaluation_result": {"is_grounded": False, "reason": reason},
            "critique_result": {"is_grounded": False, "reason": reason},
            "retry_count": iteration_count,
            "iteration_history": iteration_history,
            "evaluator_prompt": "",
        }

    context_blocks = []
    for idx, doc in enumerate(documents, start=1):
        source = str(doc.metadata.get("source", "unknown_source"))
        context_blocks.append(
            f"[DOC {idx}] Fuente: {source}\\n"
            f"Contenido: {_truncate(doc.page_content)}"
        )
    context = "\\n\\n".join(context_blocks)

    prompt = load_prompt("evaluator").format(
        question=question,
        context=context,
        answer=generation,
    )

    try:
        evaluation = _evaluator_llm().with_structured_output(GroundingEvaluation).invoke(prompt)
        is_grounded = bool(evaluation.is_grounded and evaluation.citation_compliance)
        reason = evaluation.reason.strip()
        unsupported_claims = [claim.strip() for claim in evaluation.unsupported_claims if claim.strip()]
        evaluation_result = {
            "is_grounded": bool(evaluation.is_grounded),
            "citation_compliance": bool(evaluation.citation_compliance),
            "reason": reason,
            "unsupported_claims": unsupported_claims,
        }
    except Exception as exc:
        logger.warning(
            "Evaluator LLM failed (possible rate limit or connection issue). "
            "provider=%s model=%s.",
            GROUNDING_EVALUATOR_LLM.provider,
            GROUNDING_EVALUATOR_LLM.model,
            exc_info=exc,
        )
        is_grounded = False
        reason = "No fue posible ejecutar verificacion estructurada."
        unsupported_claims = []
        evaluation_result = {
            "is_grounded": False,
            "citation_compliance": False,
            "reason": reason,
            "unsupported_claims": unsupported_claims,
        }
        # Do not retry on connection failure.
        iteration_count = max(0, _safe_int(state.get("iteration_count", 0), 0))
        iteration_history = _append_iteration_history(
            state,
            iteration_count=iteration_count,
            k_value=k_value,
            is_grounded=False,
            decision="end",
            reason=reason,
        )
        failure_reason = (
            "rate_limit_429" if is_rate_limit_429(exc) else "evaluator_connection_failure"
        )
        return {
            **state,
            "is_grounded": False,
            "evaluation_decision": "end",
            "iteration_count": iteration_count,
            "evaluation_result": evaluation_result,
            "critique_result": evaluation_result,
            "retry_count": iteration_count,
            "iteration_history": iteration_history,
            "evaluator_prompt": prompt,
            "llm_failure": True,
            "llm_failure_reason": failure_reason,
            "llm_failure_source": f"{GROUNDING_EVALUATOR_LLM.provider}:{GROUNDING_EVALUATOR_LLM.model}",
        }

    if is_grounded:
        decision = "end"
        iteration_history = _append_iteration_history(
            state,
            iteration_count=iteration_count,
            k_value=k_value,
            is_grounded=True,
            decision=decision,
            reason=reason,
        )
        return {
            **state,
            "is_grounded": True,
            "evaluation_decision": decision,
            "iteration_count": iteration_count,
            "max_iterations": max_iterations,
            "evaluation_result": evaluation_result,
            "critique_result": evaluation_result,
            "retry_count": iteration_count,
            "iteration_history": iteration_history,
            "evaluator_prompt": prompt,
        }

    if iteration_count < max_iterations:
        next_iteration = iteration_count + 1
        next_k = _clamp_k(k_value + RETRY_K_STEP)
        decision = "retry"
        iteration_history = _append_iteration_history(
            state,
            iteration_count=next_iteration,
            k_value=next_k,
            is_grounded=False,
            decision=decision,
            reason=reason,
        )
        return {
            **state,
            "is_grounded": False,
            "evaluation_decision": decision,
            "iteration_count": next_iteration,
            "max_iterations": max_iterations,
            "k_value": next_k,
            "evaluation_result": evaluation_result,
            "critique_result": evaluation_result,
            "retry_count": next_iteration,
            "iteration_history": iteration_history,
            "evaluator_prompt": prompt,
        }

    final_generation = generation
    detailed_reason = reason
    if unsupported_claims:
        detailed_reason = f"{reason} | Afirmaciones sin soporte: {', '.join(unsupported_claims[:3])}"
    final_generation = _insufficient_evidence_answer(
        question,
        detailed_reason or "No se pudo confirmar grounding completo tras agotar reintentos.",
    )
    decision = "end"
    iteration_history = _append_iteration_history(
        state,
        iteration_count=iteration_count,
        k_value=k_value,
        is_grounded=False,
        decision=decision,
        reason=detailed_reason,
    )

    return {
        **state,
        "is_grounded": False,
        "evaluation_decision": decision,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "generation": final_generation,
        "evaluation_result": evaluation_result,
        "critique_result": evaluation_result,
        "retry_count": iteration_count,
        "iteration_history": iteration_history,
        "evaluator_prompt": prompt,
    }


def route_after_evaluation(state: AgentState) -> Literal["retry", "end"]:
    """Route to retriever on retry decision, otherwise finish."""
    decision = str(state.get("evaluation_decision", "")).strip().lower()
    if decision == "retry":
        return "retry"
    return "end"
