from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency in tests
    load_dotenv = None


SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".html", ".htm")
DEFAULT_MIN_DOCS = 50
DEFAULT_DOCS_PATH = "docs"
DEFAULT_VECTORSTORE_PATH = "db/chroma_db"
REQUIRED_ENV_KEYS = ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")


@dataclass(frozen=True)
class Settings:
    docs_path: Path
    vectorstore_path: Path
    min_docs: int
    supported_extensions: tuple[str, ...]
    required_env_keys: tuple[str, ...]


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _resolve_path(value: str) -> Path:
    return Path(value).expanduser()


def load_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv()

    docs_path = _resolve_path(os.getenv("UNAL_RAG_DOCS_PATH", DEFAULT_DOCS_PATH))
    vectorstore_path = _resolve_path(
        os.getenv("UNAL_RAG_VECTORSTORE_PATH", DEFAULT_VECTORSTORE_PATH)
    )
    min_docs = _safe_int(os.getenv("UNAL_RAG_MIN_DOCS"), DEFAULT_MIN_DOCS)

    return Settings(
        docs_path=docs_path,
        vectorstore_path=vectorstore_path,
        min_docs=min_docs,
        supported_extensions=SUPPORTED_EXTENSIONS,
        required_env_keys=REQUIRED_ENV_KEYS,
    )
