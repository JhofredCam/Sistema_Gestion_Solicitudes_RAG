from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRoleConfig:
    provider: str
    model: str
    temperature: float
    rationale: str


# Fast, low-latency tasks.
ROUTER_LLM = LLMRoleConfig(
    provider="groq",
    model="llama-3.1-8b-instant",
    temperature=0.0,
    rationale="Clasificacion y enrutamiento rapido.",
)

K_SELECTOR_LLM = LLMRoleConfig(
    provider="groq",
    model="llama-3.1-8b-instant",
    temperature=0.0,
    rationale="Decision veloz de k para retrieval dinamico.",
)

DIRECT_LLM = LLMRoleConfig(
    provider="groq",
    model="llama-3.1-8b-instant",
    temperature=0.2,
    rationale="Respuestas generales sin retrieval con baja latencia.",
)

# Reasoning/grounding tasks on retrieved context.
RAG_GENERATION_LLM = LLMRoleConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    temperature=0.0,
    rationale="Sintesis controlada sobre contexto recuperado.",
)

GROUNDING_EVALUATOR_LLM = LLMRoleConfig(
    provider="gemini",
    model="gemini-2.5-flash",
    temperature=0.0,
    rationale="Verificacion semantica de soporte factual y coherencia.",
)
