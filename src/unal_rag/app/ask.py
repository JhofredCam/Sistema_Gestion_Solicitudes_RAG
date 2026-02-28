from __future__ import annotations

import sys
import json
from pathlib import Path

from ..config.settings import Settings


def _ensure_repo_root_on_path() -> None:
    # Add repo root so `import src.main` works when running CLI from repo.
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def run_ask(
    settings: Settings,
    *,
    question: str | None,
    max_iterations: int,
    trace: bool = False,
    reset_memory: bool = False,
) -> int:
    _ = settings
    if not question or not question.strip():
        print("Provide a question, e.g. `unal-rag ask \"Tu pregunta\"`.")
        return 1

    _ensure_repo_root_on_path()
    try:
        from src.main import build_workflow
    except Exception as exc:
        print(f"Failed to import workflow: {exc}")
        return 1

    if reset_memory:
        _reset_memory_storage()

    graph = build_workflow()
    result = graph.invoke(
        {
            "question": question.strip(),
            "iteration_count": 0,
            "max_iterations": max(0, int(max_iterations)),
        },
        config={"configurable": {"thread_id": "default"}},
    )

    print(result.get("generation", ""))
    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for source in sources:
            print(f"- {source}")
    if trace:
        trace_payload = {
            "intent": result.get("intent"),
            "k_value": result.get("k_value"),
            "selected_k_reason": result.get("selected_k_reason"),
            "selected_k_source": result.get("selected_k_source"),
            "retrieved_chunks": result.get("retrieval_trace", []),
            "final_prompt": result.get("final_prompt") or result.get("generator_prompt"),
            "critique_result": result.get("critique_result") or result.get("evaluation_result"),
            "retry_count": result.get("retry_count", result.get("iteration_count")),
        }
        print("\nTrace:")
        print(json.dumps(trace_payload, ensure_ascii=False, indent=2))
    return 0


def _reset_memory_storage() -> None:
    from pathlib import Path

    memory_path = Path("db") / "memory.json"
    checkpoint_path = Path("db") / "langgraph_checkpoints.sqlite"
    for path in (memory_path, checkpoint_path):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
