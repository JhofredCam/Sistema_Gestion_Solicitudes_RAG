from __future__ import annotations

from typing import Dict, List

from langchain_core.tools import tool


@tool
def calcular_promedio(notas: List[float]) -> float:
    """Calcula el promedio simple de una lista de notas."""
    if not notas:
        raise ValueError("La lista de notas no puede estar vacia.")
    return sum(notas) / len(notas)


@tool
def calcular_creditos_faltantes(creditos_requeridos: int, creditos_aprobados: int) -> int:
    """Retorna cuántos créditos faltan para cumplir el requisito."""
    if creditos_requeridos < 0 or creditos_aprobados < 0:
        raise ValueError("Los creditos no pueden ser negativos.")
    return max(0, creditos_requeridos - creditos_aprobados)


@tool
def contar_menciones_norma(texto: str, termino: str) -> int:
    """Cuenta menciones de un término en un texto (case-insensitive)."""
    if not texto or not termino:
        return 0
    return texto.lower().count(termino.lower())


@tool
def calcular_plazo(fecha_inicio: str, dias: int) -> str:
    """
    Calcula fecha límite a partir de fecha_inicio (YYYY-MM-DD) + dias.
    Devuelve YYYY-MM-DD.
    """
    from datetime import datetime, timedelta

    if dias < 0:
        raise ValueError("Dias no pueden ser negativos.")
    dt = datetime.strptime(fecha_inicio, "%Y-%m-%d")
    return (dt + timedelta(days=dias)).strftime("%Y-%m-%d")


@tool
def verificar_requisitos(perfil: Dict[str, object], requisitos: Dict[str, object]) -> Dict[str, object]:
    """
    Verifica cumplimiento de requisitos basados en perfil y umbrales.
    """
    faltantes = []
    if "min_creditos" in requisitos and "creditos_aprobados" in perfil:
        if perfil["creditos_aprobados"] < requisitos["min_creditos"]:
            faltantes.append("creditos")
    if "min_promedio" in requisitos and "promedio" in perfil:
        if perfil["promedio"] < requisitos["min_promedio"]:
            faltantes.append("promedio")
    if "max_semestres" in requisitos and "semestres" in perfil:
        if perfil["semestres"] > requisitos["max_semestres"]:
            faltantes.append("semestres")

    return {
        "cumple": len(faltantes) == 0,
        "faltantes": faltantes,
    }
