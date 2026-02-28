from __future__ import annotations

import re
from typing import Any, Dict

from ..unal_rag.memory import MemoryStore
from ..state import AgentState


_PROMEDIO_RE = re.compile(
    r"(promedio|papa)(?:\s+es(?:\s+de)?|\s*[:=]|\s+de)\s*([0-9]+(?:[.,][0-9]+)?)",
    re.IGNORECASE,
)
_CREDITOS_RE = re.compile(r"credito(?:s)?(?:\s+aprobados|\s*[:=])\s*([0-9]+)", re.IGNORECASE)
_SEMESTRE_RE = re.compile(r"semestre(?:\s+actual|\s*[:=])\s*([0-9]+)", re.IGNORECASE)
_PROGRAMA_RE = re.compile(r"programa(?:\s+es|\s*[:=])\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ\\s-]{3,})", re.IGNORECASE)
_GLOSSARY_RE = re.compile(
    r"\b([A-ZÁÉÍÓÚÜÑ]{2,})\b\s+es\s+([^\\n\\.]{4,})",
    re.IGNORECASE,
)
_PLAN_CODE_RE = re.compile(r"\b(3\\d{3})\b", re.IGNORECASE)
_MEMORY_INTENT_RE = re.compile(
    r"\b(recuerda|recordar|guarda|guardar|almacena|almacenar|memoriza|ten en cuenta)\b",
    re.IGNORECASE,
)


def _normalize_float(value: str) -> float:
    return float(value.replace(",", "."))


def _extract_profile(question: str) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    if not question:
        return profile

    promedio = _PROMEDIO_RE.search(question)
    if promedio:
        profile["promedio"] = _normalize_float(promedio.group(2))

    creditos = _CREDITOS_RE.search(question)
    if creditos:
        profile["creditos_aprobados"] = int(creditos.group(1))

    semestre = _SEMESTRE_RE.search(question)
    if semestre:
        profile["semestres"] = int(semestre.group(1))

    programa = _PROGRAMA_RE.search(question)
    if programa:
        profile["programa"] = programa.group(1).strip()

    return profile


def _extract_glossary(question: str) -> Dict[str, str]:
    glossary: Dict[str, str] = {}
    if not question:
        return glossary
    match = _GLOSSARY_RE.search(question)
    if match:
        key = match.group(1).strip().upper()
        value = match.group(2).strip()
        # Avoid treating numeric PAPA updates as glossary entries.
        if any(ch.isdigit() for ch in value):
            return glossary
        if key == "PAPA" and not any(ch.isalpha() for ch in value):
            return glossary
        glossary[key] = value
    return glossary


def _extract_plan_code(question: str) -> str | None:
    if not question:
        return None
    match = _PLAN_CODE_RE.search(question)
    if match:
        return match.group(1)
    return None


def _is_self_profile_update(question: str) -> bool:
    normalized = question.lower()
    if re.search(r"\bmi\s+(papa|promedio)\b", normalized) and re.search(r"\d", normalized):
        return True
    if re.search(r"\bmi\s+semestre\b", normalized) and re.search(r"\d", normalized):
        return True
    if re.search(r"\bmi\s+programa\b", normalized) and "es" in normalized:
        return True
    if "tengo" in normalized and ("creditos" in normalized or "semestre" in normalized):
        return True
    return False


def memory_load_node(state: AgentState) -> AgentState:
    """Load persisted memory profile from disk into state."""
    store = MemoryStore()
    memory = store.load()
    if not memory:
        memory = {
            "glossary": {
                "PAPA": "Promedio Aritmético Ponderado Acumulado",
            }
        }
        store.save(memory)
    return {**state, "memory": memory}


def memory_update_node(state: AgentState) -> AgentState:
    """Update persisted memory profile based on user message."""
    question = str(state.get("question", "")).strip()
    memory = dict(state.get("memory", {}) or {})
    memory_intent = bool(_MEMORY_INTENT_RE.search(question)) or _is_self_profile_update(
        question
    )
    memory_updated = False

    updates = _extract_profile(question)
    if updates:
        memory.update(updates)
        memory_updated = memory_intent
    glossary_updates = _extract_glossary(question)
    if glossary_updates:
        glossary = dict(memory.get("glossary", {}) or {})
        glossary.update(glossary_updates)
        memory["glossary"] = glossary
        memory_updated = memory_intent
    plan_code = _extract_plan_code(question)
    if plan_code:
        memory["plan_code"] = plan_code
        memory_updated = memory_intent
    if updates or glossary_updates or plan_code:
        MemoryStore().save(memory)

    return {**state, "memory": memory, "memory_updated": memory_updated}
