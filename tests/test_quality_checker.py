from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from enterprise_rag.quality.quality_checker import (
    analyze_page_texts,
    check_pdf_quality,
)


def test_blank_pages_are_rejected_without_counting_page_separators() -> None:
    result = analyze_page_texts(["", "", ""], threshold=0.80)

    assert result.passed is False
    assert result.error_code == "NO_RECOGNIZABLE_TEXT"
    assert result.recognizable_chars == 0
    assert result.average_chars_per_page == 0
    assert result.text_page_ratio == 0


def test_replacement_and_control_characters_are_invalid() -> None:
    result = analyze_page_texts(["abcd\ufffd\x03"], threshold=0.80)

    assert result.valid_char_ratio == pytest.approx(4 / 6)
    assert result.passed is False
    assert result.error_code == "LOW_VALID_CHAR_RATIO"


def test_exactly_eighty_percent_passes() -> None:
    result = analyze_page_texts(["abcd\ufffd"], threshold=0.80)

    assert result.valid_char_ratio == pytest.approx(0.80)
    assert result.passed is True
    assert result.error_code is None


def test_whitespace_is_excluded_from_the_ratio_and_density_is_reported() -> None:
    result = analyze_page_texts(["AB 12\n", "", "C"], threshold=0.80)

    assert result.extracted_chars == 5
    assert result.recognizable_chars == 5
    assert result.valid_char_ratio == 1.0
    assert result.average_chars_per_page == pytest.approx(5 / 3)
    assert result.text_page_ratio == pytest.approx(2 / 3)


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.save(path)
    finally:
        doc.close()


def test_check_pdf_quality_reads_only_requested_pages(tmp_path: Path) -> None:
    pdf = tmp_path / "readable.pdf"
    _write_pdf(pdf, ["Readable annual report", "Second page", "Third page"])

    result = check_pdf_quality(pdf, sample_pages=2)

    assert result.passed is True
    assert result.sample_pages == 2
    assert result.text_page_ratio == 1.0
    assert result.recognizable_chars > 0


def test_check_pdf_quality_rejects_multi_page_blank_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, ["", "", ""])

    result = check_pdf_quality(pdf)

    assert result.passed is False
    assert result.error_code == "NO_RECOGNIZABLE_TEXT"
    assert result.sample_pages == 3


def test_missing_pdf_returns_stable_error_code(tmp_path: Path) -> None:
    result = check_pdf_quality(tmp_path / "missing.pdf")

    assert result.passed is False
    assert result.error_code == "FILE_NOT_FOUND"


@pytest.mark.parametrize(
    ("sample_pages", "threshold"),
    [(0, 0.8), (-1, 0.8), (5, -0.1), (5, 1.1)],
)
def test_invalid_configuration_raises(sample_pages: int, threshold: float) -> None:
    with pytest.raises(ValueError):
        check_pdf_quality("unused.pdf", sample_pages=sample_pages, threshold=threshold)
