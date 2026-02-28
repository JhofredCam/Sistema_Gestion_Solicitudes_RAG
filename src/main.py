from __future__ import annotations

from pathlib import Path
import atexit

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    SqliteSaver = None

from langgraph.checkpoint.memory import MemorySaver

from .nodes.evaluator import evaluate_grounding_node, route_after_evaluation
from .nodes.generator import direct_llm_node, rag_generator_node
from .nodes.memory import memory_load_node, memory_update_node
from .nodes.retriever import retriever_node, select_k_node
from .nodes.router import classify_intent, route_by_intent
from .nodes.tools_pre import tools_pre_node
from .nodes.tools_post import tools_post_node
from .state import AgentState


def build_workflow():
    checkpoint_path = Path("db") / "langgraph_checkpoints.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if SqliteSaver is not None:
        cm = SqliteSaver.from_conn_string(str(checkpoint_path))
        checkpointer = cm.__enter__()
        atexit.register(cm.__exit__, None, None, None)
    else:
        checkpointer = MemorySaver()

    workflow = StateGraph(AgentState)

    workflow.add_node("memory_load", memory_load_node)
    workflow.add_node("memory_update", memory_update_node)
    workflow.add_node("tools_pre", tools_pre_node)
    workflow.add_node("intent_router", classify_intent)
    workflow.add_node("k_selector", select_k_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("tools_post", tools_post_node)
    workflow.add_node("rag_generator", rag_generator_node)
    workflow.add_node("evaluator", evaluate_grounding_node)
    workflow.add_node("direct_llm", direct_llm_node)

    workflow.add_edge(START, "memory_load")
    workflow.add_edge("memory_load", "memory_update")
    workflow.add_edge("memory_update", "tools_pre")
    workflow.add_conditional_edges(
        "tools_pre",
        lambda state: "end" if state.get("tool_handled") else "intent_router",
        {
            "end": END,
            "intent_router": "intent_router",
        },
    )
    workflow.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "k_selector": "k_selector",
            "direct_llm": "direct_llm",
        },
    )

    workflow.add_edge("k_selector", "retriever")
    workflow.add_edge("retriever", "tools_post")
    workflow.add_conditional_edges(
        "tools_post",
        lambda state: "end" if state.get("tool_handled") else "rag_generator",
        {
            "end": END,
            "rag_generator": "rag_generator",
        },
    )
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

    return workflow.compile(checkpointer=checkpointer)

