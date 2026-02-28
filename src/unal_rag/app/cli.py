from __future__ import annotations

import argparse
import sys

from ..config.settings import load_settings
from ..config.logging import configure_logging
from ..config.logging import configure_logging
from .ask import run_ask
from .doctor import run_doctor
from .ingest import run_ingest


def _handle_doctor(args: argparse.Namespace) -> int:
    settings = load_settings()
    return run_doctor(settings, strict=args.strict)


def _handle_ingest(args: argparse.Namespace) -> int:
    settings = load_settings()
    return run_ingest(
        settings,
        docs_path=args.docs_path,
        vectorstore_path=args.vectorstore_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def _handle_ask(args: argparse.Namespace) -> int:
    settings = load_settings()
    configure_logging(verbose_http=args.trace)
    return run_ask(
        settings,
        question=args.question,
        max_iterations=args.max_iterations,
        trace=args.trace,
        reset_memory=args.reset_memory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unal-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Validate docs path, env keys, and vector index."
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when docs_count is below the minimum.",
    )
    doctor_parser.set_defaults(func=_handle_doctor)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Build or update the vector index."
    )
    ingest_parser.add_argument(
        "--docs-path",
        help="Override docs path (defaults to UNAL_RAG_DOCS_PATH).",
    )
    ingest_parser.add_argument(
        "--vectorstore-path",
        help="Override vector store path (defaults to UNAL_RAG_VECTORSTORE_PATH).",
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for splitting documents.",
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=48,
        help="Chunk overlap for splitting documents.",
    )
    ingest_parser.set_defaults(func=lambda args: _handle_ingest(args))

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question to the RAG system."
    )
    ask_parser.add_argument("question", nargs="?", help="Question text.")
    ask_parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum retrieval retries for grounding.",
    )
    ask_parser.add_argument(
        "--trace",
        action="store_true",
        help="Print traceability data for debugging and audits.",
    )
    ask_parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Clear persisted memory and checkpoints before running.",
    )
    ask_parser.set_defaults(func=lambda args: _handle_ask(args))

    return parser


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
