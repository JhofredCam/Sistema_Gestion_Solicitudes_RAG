from __future__ import annotations

import re
from typing import Dict, List

from langchain_core.tools import tool


_PLAN_CODE_RE = re.compile(r"\b(3\d{3})\b")
_PLAN_LABEL_RE = re.compile(
    r"(plan(?:\s+de\s+estudios)?\s*(?:codigo|c[oó]digo)?\s*)(3\d{3})",
    re.IGNORECASE,
)
_PROGRAM_RE = re.compile(
    r"Ingenier[ií]a de Sistemas e Inform[aá]tica",
    re.IGNORECASE,
)


def _extract_plan_codes(text: str) -> List[str]:
    codes = set(_PLAN_CODE_RE.findall(text))
    labeled = set(code for _, code in _PLAN_LABEL_RE.findall(text))
    return sorted(codes | labeled)


def _question_mentions_plan(question: str, codes: List[str]) -> bool:
    normalized = question.lower()
    if any(code in normalized for code in codes):
        return True
    if "plan" in normalized and any(code in normalized for code in codes):
        return True
    return False


@tool
def clarificar_plan(contexto: str, pregunta: str) -> Dict:
    """
    Detecta planes de estudio en el contexto recuperado y decide si se requiere aclaración.
    """
    codes = _extract_plan_codes(contexto)
    program_match = bool(_PROGRAM_RE.search(contexto))
    needs_clarification = len(codes) > 1 and not _question_mentions_plan(pregunta, codes)
    return {
        "program_match": program_match,
        "plan_codes": codes,
        "needs_clarification": needs_clarification,
    }
