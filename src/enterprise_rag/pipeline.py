"""端到端编排：串起 parse → serialize → chunk → ingest → retrieve → rerank → answer。

设计原则：每个阶段独立运行、产物落盘到 data/parsed/，下游阶段可以单独重跑，
不依赖上游重算——调试 RAG 时这是刚需（原项目靠注释代码切阶段，我们不这么干）。
"""

from __future__ import annotations

from pathlib import Path

from enterprise_rag.config import DATA_DIR, load_config


class Pipeline:
    def __init__(self, config_name: str = "default") -> None:
        self.config = load_config(config_name)
        self.parsed_dir = DATA_DIR / "parsed"

    def parse(self, pdf_dir: Path | None = None) -> None:
        """阶段 1：解析 data/raw/ 下全部 PDF → data/parsed/docs/"""
        from enterprise_rag.parsing import pdf_parser

        pdf_dir = pdf_dir or DATA_DIR / "raw"
        out_dir = self.parsed_dir / "docs"
        out_dir.mkdir(parents=True, exist_ok=True)
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            doc = pdf_parser.parse_pdf(pdf, self.config)
            (out_dir / f"{pdf.stem}.json").write_text(
                __import__("json").dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[parse] {pdf.name} 完成")

    def ingest(self) -> None:
        """阶段 2：切分 + 嵌入 + 建索引 → data/parsed/index/"""
        raise NotImplementedError("Pipeline.ingest 待实现")

    def ask(self, question: str) -> str:
        """阶段 3：retrieve → rerank → answer，返回答案文本。"""
        raise NotImplementedError("Pipeline.ask 待实现")
