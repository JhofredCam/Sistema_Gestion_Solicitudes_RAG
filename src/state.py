from typing import Any, Dict, List, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # User question
    question: str

    # Classified intent: busqueda, resumen, comparacion, general
    intent: str

    # Retrieved chunks from the vector DB
    documents: List[Document]

    # LLM-generated answer
    generation: str

    # Dynamic k selected for retrieval
    k_value: int

    # Grounding verification result
    is_grounded: bool

    # Retry loop counters
    iteration_count: int
    max_iterations: int

    # Graph routing decision emitted by evaluator: retry | end
    evaluation_decision: str

    # Source traceability
    sources: List[str]

    # Retrieved chunk-level traceability
    retrieval_trace: List[Dict[str, Any]]

    # Final prompt sent to generator/evaluator
    generator_prompt: str
    evaluator_prompt: str

    # Last evaluator output for auditing
    evaluation_result: Dict[str, Any]

    # Per-iteration loop history
    iteration_history: List[Dict[str, Any]]
