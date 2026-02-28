from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader, DirectoryLoader, TextLoader

from ..config.settings import Settings

try:
    from ingestion_pipeline import create_vector_store, split_documents
except Exception as exc:  # pragma: no cover - runtime import guard
    create_vector_store = None
    split_documents = None
    _INGEST_IMPORT_ERROR = exc
else:
    _INGEST_IMPORT_ERROR = None


_HTML_EXTS = {".html", ".htm"}
_TEXT_EXTS = {".txt"}


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version() -> str:
    return os.getenv("UNAL_RAG_INGEST_VERSION", "v1")


def _load_with_loader(docs_path: Path, pattern: str, loader_cls) -> list:
    loader = DirectoryLoader(
        path=str(docs_path),
        glob=pattern,
        loader_cls=loader_cls,
        loader_kwargs={"open_encoding": "utf-8"},
    )
    return loader.load()


def _load_documents(settings: Settings, docs_path: Path) -> list:
    docs = []
    exts = {ext.lower() for ext in settings.supported_extensions}

    if exts & _HTML_EXTS:
        docs.extend(_load_with_loader(docs_path, "**/*.html", BSHTMLLoader))
        docs.extend(_load_with_loader(docs_path, "**/*.htm", BSHTMLLoader))

    if exts & _TEXT_EXTS:
        docs.extend(_load_with_loader(docs_path, "**/*.txt", TextLoader))

    return docs


def _enrich_chunks(chunks: list) -> None:
    ingested_at = _now_iso()
    version = _version()

    for idx, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk.metadata or {})
        source = str(metadata.get("source", "unknown_source"))
        source_path = Path(source)
        doc_id = metadata.get("doc_id") or _hash_text(str(source_path))
        chunk_id = metadata.get("chunk_id") or f"{doc_id}-{idx}"
        content_hash = _hash_text(chunk.page_content)

        metadata.update(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "content_hash": content_hash,
                "ingested_at": metadata.get("ingested_at", ingested_at),
                "version": metadata.get("version", version),
                "source_path": metadata.get("source_path", str(source_path)),
            }
        )

        chunk.metadata = metadata


def run_ingest(
    settings: Settings,
    *,
    docs_path: str | None,
    vectorstore_path: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    if create_vector_store is None or split_documents is None:
        print(f"Failed to import ingestion pipeline: {_INGEST_IMPORT_ERROR}")
        return 1

    docs_root = Path(docs_path).expanduser() if docs_path else settings.docs_path
    vectorstore_root = (
        Path(vectorstore_path).expanduser()
        if vectorstore_path
        else settings.vectorstore_path
    )

    if not docs_root.exists():
        print(f"Docs path not found: {docs_root}")
        return 1

    documents = _load_documents(settings, docs_root)
    if not documents:
        print(f"No supported documents found in {docs_root}")
        return 1

    chunks = split_documents(
        documents,
        chunk_size=max(1, int(chunk_size)),
        chunk_overlap=max(0, int(chunk_overlap)),
    )
    _enrich_chunks(chunks)

    create_vector_store(chunks, persist_directory=str(vectorstore_root))
    return 0
