from pathlib import Path

from unal_rag.app.doctor import build_report, count_supported_docs, run_doctor
from unal_rag.config.settings import REQUIRED_ENV_KEYS, SUPPORTED_EXTENSIONS, Settings


def test_count_supported_docs_counts_only_supported(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "one.txt").write_text("a", encoding="utf-8")
    (docs_path / "two.html").write_text("b", encoding="utf-8")
    (docs_path / "three.pdf").write_text("c", encoding="utf-8")
    (docs_path / "ignore.docx").write_text("d", encoding="utf-8")

    count = count_supported_docs(docs_path, SUPPORTED_EXTENSIONS)

    assert count == 3


def test_build_report_flags_missing_env_and_index(
    tmp_path: Path, monkeypatch
) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "one.txt").write_text("a", encoding="utf-8")

    for key in REQUIRED_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(
        docs_path=docs_path,
        vectorstore_path=tmp_path / "db" / "chroma_db",
        min_docs=2,
        supported_extensions=SUPPORTED_EXTENSIONS,
        required_env_keys=REQUIRED_ENV_KEYS,
    )

    report = build_report(settings)

    assert report.docs_path_exists is True
    assert report.docs_count == 1
    assert report.env_missing == REQUIRED_ENV_KEYS
    assert report.index_present is False


def test_run_doctor_strict_exits_nonzero(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()

    settings = Settings(
        docs_path=docs_path,
        vectorstore_path=tmp_path / "db" / "chroma_db",
        min_docs=1,
        supported_extensions=SUPPORTED_EXTENSIONS,
        required_env_keys=REQUIRED_ENV_KEYS,
    )

    exit_code = run_doctor(settings, strict=True)

    assert exit_code == 1
