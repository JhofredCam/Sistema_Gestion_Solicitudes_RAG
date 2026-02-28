from __future__ import annotations

from typing import Dict

from langchain_core.tools import tool


@tool
def verificar_perdida_calidad_estudiante(
    papa: float | None = None,
    promedio: float | None = None,
) -> Dict[str, object]:
    """
    Verifica pérdida de calidad de estudiante según regla: PAPA/Promedio < 3.0.
    Devuelve un dict con decisión y motivo.
    """
    value = papa if papa is not None else promedio
    if value is None:
        return {
            "tiene_dato": False,
            "perdio_calidad": None,
            "motivo": "No se proporcionó PAPA ni promedio.",
        }
    try:
        value = float(value)
    except (TypeError, ValueError):
        return {
            "tiene_dato": False,
            "perdio_calidad": None,
            "motivo": "Valor de PAPA/promedio inválido.",
        }

    perdio = value < 3.0
    return {
        "tiene_dato": True,
        "perdio_calidad": perdio,
        "motivo": "PAPA/promedio < 3.0" if perdio else "PAPA/promedio >= 3.0",
    }
