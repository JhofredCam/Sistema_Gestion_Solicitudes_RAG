from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes.evaluator import evaluate_grounding_node, route_after_evaluation
from .nodes.generator import direct_llm_node, rag_generator_node
from .nodes.retriever import retriever_node, select_k_node
from .nodes.router import classify_intent, route_by_intent
from .state import AgentState


def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent_router", classify_intent)
    workflow.add_node("k_selector", select_k_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("rag_generator", rag_generator_node)
    workflow.add_node("evaluator", evaluate_grounding_node)
    workflow.add_node("direct_llm", direct_llm_node)

    workflow.add_edge(START, "intent_router")
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "k_selector": "k_selector",
            "direct_llm": "direct_llm",
        },
    )

    workflow.add_edge("k_selector", "retriever")
    workflow.add_edge("retriever", "rag_generator")
    workflow.add_edge("rag_generator", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "retry": "retriever",
            "end": END,
        },
    )

    workflow.add_edge("direct_llm", END)

    return workflow.compile()

