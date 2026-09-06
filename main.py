"""CLI 入口：每个流水线阶段一个子命令。

用法：
  python main.py parse [--config default] [--pdf-dir PATH]
  python main.py ingest [--config default]
  python main.py ask "问题" [--config default]
"""

from __future__ import annotations

import typer

from enterprise_rag.pipeline import Pipeline

app = typer.Typer(help="Enterprise RAG 流水线", no_args_is_help=True)


@app.command()
def parse(
    config: str = typer.Option("default", "--config", "-c", help="configs/ 下的配置名"),
    pdf_dir: str = typer.Option(None, "--pdf-dir", help="PDF 目录，默认 data/raw/"),
) -> None:
    """阶段 1：解析 PDF → data/parsed/docs/"""
    from pathlib import Path

    pipe = Pipeline(config)
    pipe.parse(Path(pdf_dir) if pdf_dir else None)


@app.command()
def ingest(
    config: str = typer.Option("default", "--config", "-c"),
) -> None:
    """阶段 2：切分 + 嵌入 + 建索引"""
    Pipeline(config).ingest()


@app.command()
def ask(
    question: str = typer.Argument(..., help="要问的问题"),
    config: str = typer.Option("default", "--config", "-c"),
) -> None:
    """阶段 3：检索 → 重排序 → 生成答案"""
    print(Pipeline(config).ask(question))


if __name__ == "__main__":
    app()
