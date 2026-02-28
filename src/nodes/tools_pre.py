from __future__ import annotations

import re

from ..state import AgentState
from ..tools.academics import (
    calcular_creditos_faltantes,
    calcular_plazo,
    calcular_promedio,
)
from ..tools.academic_status import verificar_perdida_calidad_estudiante


def _extract_numbers(text: str) -> list[float]:
    return [
        float(x.replace(",", "."))
        for x in re.findall(r"[0-9]+(?:[.,][0-9]+)?", text)
    ]


def _extract_date(text: str) -> str | None:
    # Accept YYYY-MM-DD
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", text)
    if m:
        return m.group(0)
    # Accept DD/MM/YYYY -> convert
    m = re.search(r"\b(\d{2})/(\d{2})/(20\d{2})\b", text)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def _extract_creditos(text: str) -> tuple[int | None, int | None]:
    req = None
    ap = None
    m = re.search(r"requerid[oa]s?\s+(\d+)", text)
    if m:
        req = int(m.group(1))
    m = re.search(r"aprobad[oa]s?\s+(\d+)", text)
    if m:
        ap = int(m.group(1))
    return req, ap


def _is_personal_query(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("mi ", "mio", "mía", "mis ", "tengo", "con mi", "mi actual")):
        return True
    if "he perdido" in lowered or "he perdido calidad" in lowered:
        return True
    if "he" in lowered and "calidad de estudiante" in lowered:
        return True
    if re.search(r"\b(papa|promedio)\b", lowered) and re.search(r"\d", lowered):
        return True
    return False


def tools_pre_node(state: AgentState) -> AgentState:
    question = str(state.get("question", "")).strip()
    if not question:
        return {**state, "tool_handled": False}

    lowered = question.lower()

    # perdida de calidad de estudiante (usa PAPA/Promedio de memoria o de la pregunta)
    if (
        ("calidad de estudiante" in lowered or "perdido calidad" in lowered)
        and _is_personal_query(question)
    ):
        memory = state.get("memory", {}) or {}
        papa_value = None
        nums = _extract_numbers(question)
        if nums:
            papa_value = nums[0]
        if papa_value is None and isinstance(memory, dict):
            papa_value = memory.get("promedio")
        result = verificar_perdida_calidad_estudiante.invoke({"papa": papa_value})
        if not result.get("tiene_dato"):
            return {
                **state,
                "generation": (
                    "Necesito tu PAPA actual para verificar la perdida de calidad de estudiante."
                ),
                "tool_handled": True,
                "tool_name": "verificar_perdida_calidad_estudiante",
                "tool_result": result,
                "final_prompt": "tool: verificar_perdida_calidad_estudiante",
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
            "tool_handled": True,
            "tool_name": "verificar_perdida_calidad_estudiante",
            "tool_result": result,
            "final_prompt": "tool: verificar_perdida_calidad_estudiante",
        }

    # calcular_promedio
    if "calcular promedio" in lowered or "promedio de" in lowered:
        nums = _extract_numbers(question)
        if len(nums) >= 2:
            value = calcular_promedio.invoke({"notas": nums})
            return {
                **state,
                "generation": f"El promedio es {value:.2f}.",
                "tool_handled": True,
                "tool_name": "calcular_promedio",
                "tool_result": {"promedio": value},
                "final_prompt": "tool: calcular_promedio",
            }

    # calcular_creditos_faltantes
    if "creditos falt" in lowered or "faltan creditos" in lowered:
        req, ap = _extract_creditos(lowered)
        if req is None or ap is None:
            return {
                **state,
                "generation": (
                    "Indica creditos requeridos y creditos aprobados para calcular faltantes."
                ),
                "tool_handled": True,
                "tool_name": "calcular_creditos_faltantes",
                "tool_result": {"error": "faltan datos"},
                "final_prompt": "tool: calcular_creditos_faltantes",
            }
        faltan = calcular_creditos_faltantes.invoke(
            {"creditos_requeridos": req, "creditos_aprobados": ap}
        )
        return {
            **state,
            "generation": f"Te faltan {faltan} creditos.",
            "tool_handled": True,
            "tool_name": "calcular_creditos_faltantes",
            "tool_result": {"faltan": faltan},
            "final_prompt": "tool: calcular_creditos_faltantes",
        }

    # calcular_plazo
    if "plazo" in lowered or "fecha limite" in lowered or "fecha límite" in lowered:
        fecha = _extract_date(question)
        nums = _extract_numbers(question)
        dias = None
        if nums:
            dias = int(nums[-1])
        if not fecha or dias is None:
            return {
                **state,
                "generation": "Indica fecha de inicio (YYYY-MM-DD) y numero de dias.",
                "tool_handled": True,
                "tool_name": "calcular_plazo",
                "tool_result": {"error": "faltan datos"},
                "final_prompt": "tool: calcular_plazo",
            }
        fecha_limite = calcular_plazo.invoke({"fecha_inicio": fecha, "dias": dias})
        return {
            **state,
            "generation": f"La fecha limite es {fecha_limite}.",
            "tool_handled": True,
            "tool_name": "calcular_plazo",
            "tool_result": {"fecha_limite": fecha_limite},
            "final_prompt": "tool: calcular_plazo",
        }

    return {**state, "tool_handled": False}
