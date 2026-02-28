from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config.settings import Settings


@dataclass(frozen=True)
class DoctorReport:
    docs_path: Path
    docs_path_exists: bool
    docs_count: int
    min_docs_required: int
    env_missing: tuple[str, ...]
    index_path: Path
    index_present: bool

    @property
    def meets_doc_requirement(self) -> bool:
        return self.docs_count >= self.min_docs_required


def count_supported_docs(docs_path: Path, extensions: Sequence[str]) -> int:
    if not docs_path.exists():
        return 0
    ext_set = {ext.lower() for ext in extensions}
    count = 0
    for path in docs_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            count += 1
    return count


def find_missing_env(keys: Sequence[str]) -> tuple[str, ...]:
    missing = [key for key in keys if not os.getenv(key)]
    return tuple(missing)


def check_vector_index(index_path: Path) -> bool:
    if not index_path.exists():
        return False
    if index_path.is_file():
        return index_path.stat().st_size > 0
    return any(index_path.iterdir())


def build_report(settings: Settings) -> DoctorReport:
    docs_path = settings.docs_path
    docs_path_exists = docs_path.exists()
    docs_count = count_supported_docs(docs_path, settings.supported_extensions)
    env_missing = find_missing_env(settings.required_env_keys)
    index_present = check_vector_index(settings.vectorstore_path)

    return DoctorReport(
        docs_path=docs_path,
        docs_path_exists=docs_path_exists,
        docs_count=docs_count,
        min_docs_required=settings.min_docs,
        env_missing=env_missing,
        index_path=settings.vectorstore_path,
        index_present=index_present,
    )


def format_report(report: DoctorReport) -> str:
    docs_path_display = (
        str(report.docs_path.resolve())
        if report.docs_path_exists
        else str(report.docs_path)
    )
    lines = [
        f"docs_path: {docs_path_display}",
        f"docs_count: {report.docs_count}",
        f"index_path: {report.index_path}",
        f"index_present: {'yes' if report.index_present else 'no'}",
        f"missing_env: {', '.join(report.env_missing) if report.env_missing else 'none'}",
    ]

    if not report.docs_path_exists:
        lines.append("WARNING: docs_path does not exist.")
    if not report.meets_doc_requirement:
        lines.append(
            f"WARNING: docs_count below minimum ({report.min_docs_required})."
        )
    if report.env_missing:
        lines.append("WARNING: missing env vars detected.")
    if not report.index_present:
        lines.append("WARNING: vector index not found.")

    return "\n".join(lines)


def run_doctor(settings: Settings, *, strict: bool = False) -> int:
    report = build_report(settings)
    print(format_report(report))

    if strict and not report.meets_doc_requirement:
        return 1
    return 0
