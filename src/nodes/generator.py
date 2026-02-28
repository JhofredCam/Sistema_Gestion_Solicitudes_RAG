from __future__ import annotations

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..llm_config import DIRECT_LLM, RAG_GENERATION_LLM
from ..prompt_loader import load_prompt
from ..state import AgentState
from ..tools.plan import clarificar_plan
from ..tools.academic_status import verificar_perdida_calidad_estudiante


load_dotenv()
MAX_QUOTE_CHARS = 220
MAX_CONTEXT_CHARS = 1800


def _direct_llm() -> ChatGroq:
    return ChatGroq(model=DIRECT_LLM.model, temperature=DIRECT_LLM.temperature)


def _rag_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=RAG_GENERATION_LLM.model,
        temperature=RAG_GENERATION_LLM.temperature,
    )


def _source_from_doc(doc: Document) -> str:
    return str(doc.metadata.get("source", "unknown_source"))


def _quote_from_doc(doc: Document, max_chars: int = MAX_QUOTE_CHARS) -> str:
    text = " ".join(doc.page_content.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _truncate(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _insufficient_evidence_answer() -> str:
    return (
        "Evidencia insuficiente para responder con certeza usando solo el contexto recuperado.\n"
        "Reformula la consulta o solicita mas contexto normativo especifico."
    )


def _glossary_block(memory: dict) -> str:
    glossary = memory.get("glossary") if isinstance(memory, dict) else None
    if not glossary:
        return ""
    lines = ["Glosario del usuario:"]
    for key, value in glossary.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _traceability_block(retrieval_trace: list[dict]) -> str:
    if not retrieval_trace:
        return "Trazabilidad:\n- No hubo documentos recuperados."

    lines = ["Trazabilidad:", "Documentos recuperados:"]
    for item in retrieval_trace:
        lines.append(
            "- "
            f"rank={item.get('rank')} | doc_id={item.get('doc_id')} | "
            f"chunk_id={item.get('chunk_id')} | page={item.get('page')} | "
            f"source={item.get('source')}"
        )
        lines.append(f"  fragmento: {item.get('snippet')}")
    return "\n".join(lines)


def _extract_numeric_value(text: str) -> float | None:
    import re

    match = re.search(r"([0-9]+(?:[.,][0-9]+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _is_quality_loss_query(text: str) -> bool:
    lowered = text.lower()
    return "calidad" in lowered and ("perdi" in lowered or "perd" in lowered)


class GroundedClaim(BaseModel):
    claim: str = Field(description="Afirmacion puntual derivada del contexto.")
    support_doc_ids: list[int] = Field(
        default_factory=list,
        description="Lista de doc_ids [DOC n] que respaldan la afirmacion.",
    )


class GroundedResponse(BaseModel):
    answer: str = Field(description="Respuesta en lenguaje natural.")
    insufficient_evidence: bool = Field(
        description="True si no hay suficiente evidencia para responder con certeza."
    )
    claims: list[GroundedClaim] = Field(
        default_factory=list,
        description="Afirmaciones clave de la respuesta con sus doc_ids de soporte.",
    )


def direct_llm_node(state: AgentState) -> AgentState:
    """Answer directly with an LLM, without retrieval context."""
    question = state.get("question", "").strip()
    if not question:
        return {**state, "generation": "No recibi una pregunta para responder.", "sources": []}
    if _is_quality_loss_query(question):
        memory = state.get("memory", {}) or {}
        papa_value = _extract_numeric_value(question)
        if papa_value is None and isinstance(memory, dict):
            papa_value = memory.get("promedio")
        result = verificar_perdida_calidad_estudiante.invoke({"papa": papa_value})
        if not result.get("tiene_dato"):
            return {
                **state,
                "generation": (
                    "Necesito tu PAPA actual para verificar la perdida de calidad de estudiante."
                ),
                "sources": [],
                "generator_prompt": "",
                "final_prompt": "",
            }
        perdio = result.get("perdio_calidad")
        if perdio:
            msg = (
                f"Si. Con PAPA {papa_value:.2f} (< 3.0) has perdido la calidad de estudiante."
            )
        else:
            msg = (
                f"No. Con PAPA {papa_value:.2f} (>= 3.0) no has perdido la calidad de estudiante."
            )
        return {
            **state,
            "generation": msg,
            "sources": [],
            "generator_prompt": "",
            "final_prompt": "",
        }
    if state.get("memory_updated"):
        return {
            **state,
            "generation": "Listo. He guardado esa informacion en tu perfil.",
            "sources": [],
            "generator_prompt": "",
            "final_prompt": "",
        }
    if _is_quality_loss_query(question):
        memory = state.get("memory", {}) or {}
        papa_value = None
        if isinstance(memory, dict):
            papa_value = memory.get("promedio")
        if papa_value is None:
            papa_value = _extract_numeric_value(question)
        result = verificar_perdida_calidad_estudiante.invoke({"papa": papa_value})
        if not result.get("tiene_dato"):
            return {
                **state,
                "generation": (
                    "Necesito tu PAPA actual para verificar la pérdida de calidad de estudiante."
                ),
                "sources": [],
                "generator_prompt": "",
                "final_prompt": "",
            }
        perdio = result.get("perdio_calidad")
        if perdio:
            msg = (
                f"Sí. Con PAPA {papa_value:.2f} (< 3.0) has perdido la calidad de estudiante."
            )
        else:
            msg = (
                f"No. Con PAPA {papa_value:.2f} (≥ 3.0) no has perdido la calidad de estudiante."
            )
        return {
            **state,
            "generation": msg,
            "sources": [],
            "generator_prompt": "",
            "final_prompt": "",
        }
    glossary_block = _glossary_block(state.get("memory", {}))
    question_with_glossary = (
        f"{question}\n\n{glossary_block}" if glossary_block else question
    )
    prompt = load_prompt("direct_llm").format(question=question_with_glossary)
    response = _direct_llm().invoke(prompt)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {
        **state,
        "generation": answer,
        "sources": [],
        "generator_prompt": prompt,
        "final_prompt": prompt,
    }


def rag_generator_node(state: AgentState) -> AgentState:
    """Generate grounded answer with explicit per-claim citations."""
    question = state.get("question", "").strip()
    documents = state.get("documents", [])
    retrieval_trace = state.get("retrieval_trace", [])
    if not question:
        return {**state, "generation": "No recibi una pregunta para responder."}
    if not documents:
        return {
            **state,
            "generation": _insufficient_evidence_answer(),
            "sources": [],
            "generator_prompt": "",
            "final_prompt": "",
        }

    doc_map: dict[int, Document] = {}
    context_blocks = []
    for idx, doc in enumerate(documents, start=1):
        doc_map[idx] = doc
        source = _source_from_doc(doc)
        chunk_id = doc.metadata.get("chunk_id", "unknown_chunk")
        doc_id = doc.metadata.get("doc_id", source)
        context_blocks.append(
            f"[DOC {idx}] source={source} | doc_id={doc_id} | chunk_id={chunk_id}\n"
            f"Contenido:\n{_truncate(doc.page_content)}"
        )
    context = "\n\n".join(context_blocks)

    # Clarify plan if multiple plan codes appear in retrieved context.
    memory = state.get("memory", {}) or {}
    question_for_plan = question
    if isinstance(memory, dict) and memory.get("plan_code"):
        question_for_plan = f"{question}\nPlan: {memory.get('plan_code')}"
    try:
        plan_check = clarificar_plan.invoke(
            {"contexto": context, "pregunta": question_for_plan}
        )
    except Exception:
        plan_check = {"needs_clarification": False, "plan_codes": []}
    if plan_check.get("needs_clarification"):
        plan_codes = plan_check.get("plan_codes", [])
        plan_list = ", ".join(plan_codes) if plan_codes else "N/D"
        clarification = (
            "Necesito aclarar el plan de estudios para responder con precisión.\n"
            f"En los documentos aparecen varios planes: {plan_list}.\n"
            "Indica el plan (por ejemplo, 3306 o 3302) y continúo."
        )
        return {
            **state,
            "generation": clarification,
            "sources": list(dict.fromkeys(state.get("sources", []))),
            "generator_prompt": "",
            "final_prompt": "",
        }

    intent = str(state.get("intent", "busqueda")).strip().lower()
    prompt_name = {"resumen": "rag_summary", "comparacion": "rag_compare"}.get(intent, "rag_answer")
    glossary_block = _glossary_block(state.get("memory", {}))
    question_with_glossary = (
        f"{question}\n\n{glossary_block}" if glossary_block else question
    )
    prompt = load_prompt(prompt_name).format(question=question_with_glossary, context=context)

    try:
        parsed = _rag_llm().with_structured_output(GroundedResponse).invoke(prompt)
    except Exception:
        return {
            **state,
            "generation": _insufficient_evidence_answer(),
            "sources": [],
            "generator_prompt": prompt,
            "final_prompt": prompt,
        }

    validated_claims = []
    used_doc_ids: set[int] = set()
    for claim in parsed.claims:
        valid_ids = [doc_id for doc_id in claim.support_doc_ids if doc_id in doc_map]
        if claim.claim.strip() and valid_ids:
            validated_claims.append((claim.claim.strip(), valid_ids))
            used_doc_ids.update(valid_ids)

    if parsed.insufficient_evidence or not validated_claims:
        return {
            **state,
            "generation": _insufficient_evidence_answer(),
            "sources": [],
            "generator_prompt": prompt,
            "final_prompt": prompt,
        }

    claim_lines = [
        f"- {claim_text} [DOC {', DOC '.join(str(doc_id) for doc_id in doc_ids)}]"
        for claim_text, doc_ids in validated_claims
    ]

    quote_lines = []
    sources = []
    for doc_id in sorted(used_doc_ids):
        doc = doc_map[doc_id]
        source = _source_from_doc(doc)
        sources.append(source)
        quote_lines.append(f'> [DOC {doc_id}] "{_quote_from_doc(doc)}" (source: {source})')

    final_answer = (
        f"{parsed.answer.strip()}\n\n"
        "Afirmaciones con soporte:\n"
        f"{chr(10).join(claim_lines)}\n\n"
        "Citas:\n"
        f"{chr(10).join(quote_lines)}\n\n"
        f"{_traceability_block(retrieval_trace)}"
    )

    return {
        **state,
        "generation": final_answer,
        "sources": list(dict.fromkeys(sources)),
        "generator_prompt": prompt,
        "final_prompt": prompt,
    }
