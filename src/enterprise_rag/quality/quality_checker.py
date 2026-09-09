"""PDF 文本层质量预检。

在 MinerU 解析前用 PyMuPDF 抽取前几页文本，计算有效字符率和文本密度，
将明显无文本层或包含大量异常字符的文档挡在知识库入口之外。
"""

from __future__ import annotations

import argparse
import dataclasses
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pymupdf

DEFAULT_SAMPLE_PAGES = 5
DEFAULT_THRESHOLD = 0.80


@dataclass(frozen=True)
class QualityCheckResult:
    """可直接序列化给 CLI 或 Dashboard 的质量检查结果。"""

    passed: bool
    valid_char_ratio: float
    extracted_chars: int
    recognizable_chars: int
    sample_pages: int
    average_chars_per_page: float
    text_page_ratio: float
    reason: str | None = None
    error_code: str | None = None

    @property
    def total_chars(self) -> int:
        """兼容旧调用方：非空白提取字符总数。"""
        return self.extracted_chars

    @property
    def valid_chars(self) -> int:
        """兼容旧调用方：可识别字符数。"""
        return self.recognizable_chars

    def as_dict(self) -> dict:
        """返回适合 JSON 序列化的字典。"""
        return dataclasses.asdict(self)


def _validate_options(sample_pages: int, threshold: float) -> None:
    if sample_pages <= 0:
        raise ValueError("sample_pages 必须大于 0")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")


def _is_recognizable(char: str) -> bool:
    """判断一个非空白字符是否适合作为可检索文本。"""
    if char == "\ufffd":
        return False
    category = unicodedata.category(char)
    if category.startswith("C"):
        return False
    return category[0] in {"L", "N", "P", "S"}


def analyze_page_texts(
    page_texts: Iterable[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> QualityCheckResult:
    """从逐页文本计算质量指标，不引入虚假的页间分隔字符。"""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须位于 [0, 1]")

    pages = list(page_texts)
    extracted = 0
    recognizable = 0
    text_pages = 0
    for text in pages:
        page_extracted = [char for char in text if not char.isspace()]
        page_recognizable = sum(_is_recognizable(char) for char in page_extracted)
        extracted += len(page_extracted)
        recognizable += page_recognizable
        if page_recognizable:
            text_pages += 1

    page_count = len(pages)
    ratio = recognizable / extracted if extracted else 0.0
    density = recognizable / page_count if page_count else 0.0
    text_page_ratio = text_pages / page_count if page_count else 0.0

    common = {
        "valid_char_ratio": ratio,
        "extracted_chars": extracted,
        "recognizable_chars": recognizable,
        "sample_pages": page_count,
        "average_chars_per_page": density,
        "text_page_ratio": text_page_ratio,
    }
    if recognizable == 0:
        return QualityCheckResult(
            passed=False,
            reason=f"前 {page_count} 页未提取到可识别文本",
            error_code="NO_RECOGNIZABLE_TEXT",
            **common,
        )
    if ratio < threshold:
        return QualityCheckResult(
            passed=False,
            reason=(
                f"有效字符率 {ratio:.1%} 低于阈值 {threshold:.0%}"
                f"（前 {page_count} 页，{recognizable}/{extracted} 字符）"
            ),
            error_code="LOW_VALID_CHAR_RATIO",
            **common,
        )
    return QualityCheckResult(passed=True, **common)


def _failure(code: str, reason: str) -> QualityCheckResult:
    return QualityCheckResult(
        passed=False,
        valid_char_ratio=0.0,
        extracted_chars=0,
        recognizable_chars=0,
        sample_pages=0,
        average_chars_per_page=0.0,
        text_page_ratio=0.0,
        reason=reason,
        error_code=code,
    )


def check_pdf_quality(
    pdf_path: str | Path,
    sample_pages: int = DEFAULT_SAMPLE_PAGES,
    threshold: float = DEFAULT_THRESHOLD,
) -> QualityCheckResult:
    """快速检查 PDF 前几页的文本层质量。"""
    _validate_options(sample_pages, threshold)
    path = Path(pdf_path)
    if not path.exists():
        return _failure("FILE_NOT_FOUND", f"文件不存在：{path}")

    try:
        doc = pymupdf.open(path)
    except (OSError, RuntimeError, ValueError) as exc:
        return _failure("PDF_OPEN_FAILED", f"无法打开 PDF：{exc}")

    try:
        actual_pages = min(sample_pages, len(doc))
        if actual_pages == 0:
            return _failure("PDF_NO_PAGES", "PDF 无页面")
        texts = [doc[index].get_text("text") for index in range(actual_pages)]
        return analyze_page_texts(texts, threshold)
    except (OSError, RuntimeError, ValueError) as exc:
        return _failure("TEXT_EXTRACTION_FAILED", f"PDF 文本提取失败：{exc}")
    finally:
        doc.close()


def check_and_report(
    pdf_path: Path,
    sample_pages: int = DEFAULT_SAMPLE_PAGES,
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    """检查并打印单个 PDF 的质量报告。"""
    result = check_pdf_quality(pdf_path, sample_pages, threshold)
    if result.passed:
        print(
            f"[PASS] {pdf_path.name}: 有效字符率 {result.valid_char_ratio:.1%}，"
            f"平均 {result.average_chars_per_page:.1f} 字符/页，"
            f"文本页 {result.text_page_ratio:.1%}（前 {result.sample_pages} 页）"
        )
    else:
        print(f"[FAIL] {pdf_path.name}: {result.reason}")
    return result.passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF 文档质量预检（提取前 N 页，计算有效字符率和文本密度）"
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDF 文件或目录")
    parser.add_argument("--sample-pages", type=int, default=DEFAULT_SAMPLE_PAGES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    paths: list[Path] = []
    for path in args.pdfs:
        paths.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    passed = sum(
        check_and_report(path, args.sample_pages, args.threshold) for path in paths
    )
    print(f"\n# 完成：{passed} 通过，{len(paths) - passed} 失败，共 {len(paths)} 份")


if __name__ == "__main__":
    main()
