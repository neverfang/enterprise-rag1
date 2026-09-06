"""PDF 解析：原始 PDF → 结构化文档。

用 Docling 提取标题层级、段落、表格，输出统一的中间格式（JSON）落盘到
data/parsed/docs/，供下游切分/索引使用。

TODO(实现):
- docling DocumentConverter 解析，table_mode 取自 config["parsing"]
- 统一中间格式，建议字段：
  {doc_id, source_pdf, elements: [{type: heading|paragraph|table, level, text, page}]}
- 表格以 Markdown 形式存入 text，原始结构留在 extra
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_pdf(pdf_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """解析单个 PDF，返回结构化文档（中间格式）。"""
    raise NotImplementedError("pdf_parser.parse_pdf 待实现")
