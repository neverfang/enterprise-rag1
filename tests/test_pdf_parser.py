from __future__ import annotations

import json
from pathlib import Path

from enterprise_rag.parsing import pdf_parser
from enterprise_rag.quality import QualityCheckResult


def _quality(passed: bool) -> QualityCheckResult:
    return QualityCheckResult(
        passed=passed,
        valid_char_ratio=1.0 if passed else 0.5,
        extracted_chars=100,
        recognizable_chars=100 if passed else 50,
        sample_pages=5,
        average_chars_per_page=20 if passed else 10,
        text_page_ratio=1.0,
        reason=None if passed else "有效字符率过低",
        error_code=None if passed else "LOW_VALID_CHAR_RATIO",
    )


def test_quality_rejection_returns_dashboard_contract(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not read because quality is patched")
    out_dir = tmp_path / "out"
    parse_called = False

    monkeypatch.setattr(pdf_parser, "_load_company_lookup", dict)
    monkeypatch.setattr(pdf_parser, "check_pdf_quality", lambda _: _quality(False), raising=False)

    def unexpected_parse(*args, **kwargs):
        nonlocal parse_called
        parse_called = True
        raise AssertionError("MinerU must not run for a rejected PDF")

    monkeypatch.setattr(pdf_parser, "parse_pdf", unexpected_parse)

    results = pdf_parser.parse_and_export([pdf], out_dir)

    assert parse_called is False
    assert len(results) == 1
    assert results[0].status == "rejected"
    assert results[0].error_code == "DOCUMENT_QUALITY_REJECTED"
    assert results[0].message == "该文档质量不达标，请检查后重新上传"
    assert results[0].quality["passed"] is False
    assert not (out_dir / "bad.json").exists()


def test_accepted_pdf_writes_json_and_returns_output_path(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = tmp_path / "good.pdf"
    pdf.write_bytes(b"pdf")
    out_dir = tmp_path / "out"
    report = {
        "metainfo": {
            "num_pages": 1,
            "num_tables": 0,
            "num_elements": 1,
            "parse_seconds": 0.1,
        },
        "content": [],
        "tables": [],
    }
    monkeypatch.setattr(pdf_parser, "_load_company_lookup", dict)
    monkeypatch.setattr(pdf_parser, "check_pdf_quality", lambda _: _quality(True), raising=False)
    monkeypatch.setattr(pdf_parser, "parse_pdf", lambda *args, **kwargs: report)

    results = pdf_parser.parse_and_export([pdf], out_dir)

    assert results[0].status == "accepted"
    assert results[0].error_code is None
    assert results[0].output_path == str(out_dir / "good.json")
    assert json.loads((out_dir / "good.json").read_text(encoding="utf-8")) == report


def test_existing_output_is_skipped_before_quality_check(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = tmp_path / "existing.pdf"
    pdf.write_bytes(b"pdf")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "existing.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pdf_parser, "_load_company_lookup", dict)

    def unexpected_quality(_):
        raise AssertionError("existing outputs must retain restart semantics")

    monkeypatch.setattr(pdf_parser, "check_pdf_quality", unexpected_quality, raising=False)

    results = pdf_parser.parse_and_export([pdf], out_dir)

    assert results[0].status == "skipped"
    assert results[0].output_path == str(out_dir / "existing.json")


def test_parser_exception_becomes_failed_result(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(pdf_parser, "_load_company_lookup", dict)
    monkeypatch.setattr(pdf_parser, "check_pdf_quality", lambda _: _quality(True), raising=False)

    def fail_parse(*args, **kwargs):
        raise RuntimeError("mineru failed")

    monkeypatch.setattr(pdf_parser, "parse_pdf", fail_parse)

    results = pdf_parser.parse_and_export([pdf], tmp_path / "out")

    assert results[0].status == "failed"
    assert results[0].error_code == "PDF_PARSE_FAILED"
    assert "mineru failed" in results[0].message


def test_disabled_quality_check_bypasses_checker(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "unchecked.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(pdf_parser, "_load_company_lookup", dict)

    def unexpected_quality(_):
        raise AssertionError("quality checker should be disabled")

    monkeypatch.setattr(pdf_parser, "check_pdf_quality", unexpected_quality, raising=False)
    monkeypatch.setattr(
        pdf_parser,
        "parse_pdf",
        lambda *args, **kwargs: {
            "metainfo": {
                "num_pages": 1,
                "num_tables": 0,
                "num_elements": 0,
                "parse_seconds": 0,
            },
            "content": [],
            "tables": [],
        },
    )

    results = pdf_parser.parse_and_export(
        [pdf], tmp_path / "out", quality_check=False
    )

    assert results[0].status == "accepted"
    assert results[0].quality is None
