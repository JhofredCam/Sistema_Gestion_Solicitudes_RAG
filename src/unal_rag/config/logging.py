from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO, *, verbose_http: bool = False) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not verbose_http:
        for noisy_logger in (
            "httpx",
            "google_genai",
            "sentence_transformers",
            "chromadb",
        ):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)
