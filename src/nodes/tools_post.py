from __future__ import annotations

import re
from typing import Dict

from ..state import AgentState
from ..tools.academics import contar_menciones_norma, verificar_requisitos
from ..tools.summary import resumir_norma


def _build_context(documents: list) -> str:
    blocks = []
    for idx, doc in enumerate(documents, start=1):
        blocks.append(f"[DOC {idx}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _extract_term(question: str) -> str | None:
    m = re.search(r"\"([^\"]{3,})\"", question)
    if m:
        return m.group(1)
    m = re.search(r"menciones de ([a-zA-ZÁÉÍÓÚÜÑáéíóúüñ\\s-]{3,})", question, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _extract_requisitos(context: str) -> Dict[str, object]:
    requisitos: Dict[str, object] = {}
    m = re.search(r"promedio\\s+m[ií]nimo\\s+([0-9]+(?:[.,][0-9]+)?)", context, re.IGNORECASE)
    if m:
        requisitos["min_promedio"] = float(m.group(1).replace(",", "."))
    m = re.search(r"m[ií]nimo\\s+(\\d+)\\s+cr[eé]ditos", context, re.IGNORECASE)
    if m:
        requisitos["min_creditos"] = int(m.group(1))
    m = re.search(r"m[aá]ximo\\s+(\\d+)\\s+semestres", context, re.IGNORECASE)
    if m:
        requisitos["max_semestres"] = int(m.group(1))
    return requisitos


def tools_post_node(state: AgentState) -> AgentState:
    question = str(state.get("question", "")).strip()
    intent = str(state.get("intent", "")).strip().lower()
    documents = state.get("documents", [])
    if not question or not documents:
        return {**state, "tool_handled": False}

    context = _build_context(documents)
    lowered = question.lower()

    if "menciones" in lowered:
        term = _extract_term(question)
        if not term:
            return {
                **state,
                "generation": "Indica el termino exacto a contar, por ejemplo: \"cancelacion\".",
                "tool_handled": True,
                "tool_name": "contar_menciones_norma",
                "tool_result": {"error": "faltan datos"},
                "final_prompt": "tool: contar_menciones_norma",
            }
        count = contar_menciones_norma.invoke({"texto": context, "termino": term})
        return {
            **state,
            "generation": f'El termino "{term}" aparece {count} veces en el contexto recuperado.',
            "tool_handled": True,
            "tool_name": "contar_menciones_norma",
            "tool_result": {"termino": term, "menciones": count},
            "final_prompt": "tool: contar_menciones_norma",
        }

    if intent == "resumen":
        summary = resumir_norma.invoke({"contexto": context, "pregunta": question})
        return {
            **state,
            "generation": summary,
            "tool_handled": True,
            "tool_name": "resumir_norma",
            "tool_result": {"resumen": summary},
            "final_prompt": "tool: resumir_norma",
        }

    if "cumplo requisitos" in lowered or "cumplir requisitos" in lowered:
        requisitos = _extract_requisitos(context)
        memory = state.get("memory", {}) or {}
        perfil = {
            "creditos_aprobados": memory.get("creditos_aprobados"),
            "promedio": memory.get("promedio"),
            "semestres": memory.get("semestres"),
        }
        if not requisitos or all(v is None for v in perfil.values()):
            return {
                **state,
                "generation": "No tengo suficientes datos de requisitos o tu perfil para verificar.",
                "tool_handled": True,
                "tool_name": "verificar_requisitos",
                "tool_result": {"error": "faltan datos"},
                "final_prompt": "tool: verificar_requisitos",
            }
        result = verificar_requisitos.invoke({"perfil": perfil, "requisitos": requisitos})
        if result.get("cumple"):
            msg = "Cumples los requisitos con la informacion disponible."
        else:
            faltantes = ", ".join(result.get("faltantes", []))
            msg = f"No cumples los requisitos. Faltantes: {faltantes}."
        return {
            **state,
            "generation": msg,
            "tool_handled": True,
            "tool_name": "verificar_requisitos",
            "tool_result": result,
            "final_prompt": "tool: verificar_requisitos",
        }

    return {**state, "tool_handled": False}
