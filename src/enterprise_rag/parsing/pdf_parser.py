"""PDF 解析：原始 PDF → 结构化 JSON（metainfo + 顺序内容流 + 表格）。

解析引擎：MinerU pipeline 后端（用户选型，原项目用的是 Docling）。
- 通过 MinerU CLI（mineru -p ... -o ... -b pipeline）而非其 Python API，
  CLI 在版本间更稳定
- 消费 MinerU 输出的 {name}_content_list.json，转成项目统一 JSON：
    metainfo: 文档统计 + 公司名（来自 data/pdf_metadata.csv）+ 文件 sha1
    content:  按阅读顺序的元素流 [{page, type, text, level, table_id?}]
    tables:   表格原文（HTML + 标题），线性化留给路线图第 3 步
- MinerU 的 page_idx 是 0 基，统一转成 1 基（与 PDF 阅读器页码一致）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from enterprise_rag.quality import check_pdf_quality

# src/enterprise_rag/parsing/pdf_parser.py → 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

# 每个 PDF 的解析时限（CPU 下大文件可能要几分钟）
PER_PDF_TIMEOUT = 20 * 60
QUALITY_REJECTION_MESSAGE = "该文档质量不达标，请检查后重新上传"


@dataclass(frozen=True)
class IngestionResult:
    """单份 PDF 的解析结果，可直接交给 Dashboard 展示。"""

    source: str
    status: str
    message: str
    error_code: str | None = None
    quality: dict | None = None
    output_path: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _mineru_cmd() -> list[str]:
    """定位 MinerU CLI（venv 内可直接跑本模块，PATH 里未必有 mineru）。"""
    exe = shutil.which("mineru")
    if exe:
        return [exe]
    sibling = Path(sys.executable).parent / ("mineru.exe" if os.name == "nt" else "mineru")
    if sibling.exists():
        return [str(sibling)]
    return [sys.executable, "-m", "mineru"]


def _run_mineru(pdf_path: Path, work_dir: Path) -> list[dict]:
    """对单个 PDF 调 MinerU，返回 content_list（list[dict]）。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_mineru_cmd(),
        "-p", str(pdf_path),
        "-o", str(work_dir),
        "-b", "pipeline",
        "-l", "en",
    ]
    env = {**os.environ, "MINERU_MODEL_SOURCE": "modelscope"}
    proc = subprocess.run(
        cmd, capture_output=True, timeout=PER_PDF_TIMEOUT, env=env, check=False,
        encoding="utf-8", errors="replace",  # MinerU 输出 UTF-8，中文 Windows 默认 GBK 会解码失败
    )
    if proc.returncode != 0:
        raise RuntimeError(f"MinerU 退出码 {proc.returncode}: {proc.stderr[-2000:]}")
    # 输出目录结构随版本有差异（{name}/{backend}/...），按文件名兜底搜索
    matches = list(work_dir.rglob("*_content_list.json"))
    if not matches:
        raise RuntimeError(f"未找到 content_list.json，MinerU 输出: {proc.stdout[-2000:]}")
    # 取本次 PDF 对应的（work_dir 按单 PDF 隔离，取最新的即可）
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


def _convert(content_list: list[dict], *, doc_id: str, source_pdf: str,
             company: str | None, sha1: str, seconds: float) -> dict:
    """content_list → 项目统一 JSON。"""
    content: list[dict] = []
    tables: list[dict] = []
    n_pictures = 0

    for raw in content_list:
        rtype = raw.get("type")
        if rtype == "discarded":  # MinerU 标记的噪音（页眉页脚等），不进内容流
            continue
        page = raw.get("page_idx")
        page = page + 1 if page is not None else None  # 0 基 → 1 基

        if rtype == "text":
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            level = raw.get("text_level") or 0
            content.append({
                "page": page,
                "type": "heading" if level else "paragraph",
                "text": text,
                "level": level,
            })
        elif rtype == "table":
            tid = f"t{len(tables)}"
            caption = " ".join(raw.get("table_caption") or []).strip()
            tables.append({
                "id": tid,
                "page": page,
                "html": raw.get("table_body") or "",
                "caption": caption,
            })
            content.append({"page": page, "type": "table", "text": caption, "table_id": tid, "level": 0})
        elif rtype == "image":
            n_pictures += 1
            caption = " ".join(raw.get("image_caption") or []).strip()
            content.append({"page": page, "type": "picture", "text": caption, "level": 0})
        else:  # equation 及未来新增类型
            text = (raw.get("text") or "").strip()
            if text:
                content.append({"page": page, "type": rtype or "unknown", "text": text, "level": 0})

    num_pages = max((c["page"] for c in content if c["page"]), default=0)
    return {
        "metainfo": {
            "doc_id": doc_id,
            "source_pdf": source_pdf,
            "company": company,
            "sha1": sha1,
            "engine": "mineru-pipeline",
            "num_pages": num_pages,
            "num_elements": len(content),
            "num_tables": len(tables),
            "num_pictures": n_pictures,
            "parse_seconds": round(seconds, 1),
        },
        "content": content,
        "tables": tables,
    }


def _load_company_lookup() -> dict[str, str]:
    """文件名 → 公司名（data/pdf_metadata.csv，导出自带 BOM，用 utf-8-sig）。"""
    lookup: dict[str, str] = {}
    with open(DATA_DIR / "pdf_metadata.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lookup[Path(row["filename"]).name] = row["company"]
    return lookup


def parse_pdf(pdf_path: Path, work_dir: Path, company: str | None = None) -> dict:
    """解析单个 PDF，返回统一 JSON dict。"""
    pdf_path = Path(pdf_path)
    t0 = time.perf_counter()
    content_list = _run_mineru(pdf_path, work_dir)
    return _convert(
        content_list,
        doc_id=pdf_path.stem,
        source_pdf=pdf_path.name,
        company=company,
        sha1=_sha1(pdf_path),
        seconds=time.perf_counter() - t0,
    )


def parse_and_export(pdf_paths: list[Path], out_dir: Path,
                     work_dir: Path | None = None,
                     quality_check: bool = True) -> list[IngestionResult]:
    """批量解析并落盘到 out_dir/{stem}.json（已存在的跳过，可断点续跑）。

    Args:
        pdf_paths: PDF 文件列表
        out_dir: 输出目录
        work_dir: MinerU 工作目录
        quality_check: 是否在解析前做质量预检（默认 True）
    """
    out_dir = Path(out_dir)
    work_dir = Path(work_dir) if work_dir else out_dir.parent / "mineru_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = _load_company_lookup()
    results: list[IngestionResult] = []

    for pdf in pdf_paths:
        pdf = Path(pdf)
        out = out_dir / f"{pdf.stem}.json"
        if out.exists():
            print(f"[SKIP] {pdf.name}（已解析）")
            results.append(IngestionResult(
                source=str(pdf), status="skipped", message="文档已解析，跳过",
                output_path=str(out),
            ))
            continue

        # 质量预检（解析前拦截低质量文档）
        quality: dict | None = None
        if quality_check:
            try:
                result = check_pdf_quality(pdf)
            except Exception as exc:  # noqa: BLE001 - 批处理需隔离单文件失败
                message = f"质量预检失败：{exc}"
                print(f"[FAIL] {pdf.name}: {message}")
                results.append(IngestionResult(
                    source=str(pdf), status="failed", message=message,
                    error_code="QUALITY_CHECK_FAILED",
                ))
                continue
            quality = result.as_dict()
            if not result.passed:
                print(f"[FAIL] {pdf.name}: {result.reason}")
                print(f"       {QUALITY_REJECTION_MESSAGE}")
                results.append(IngestionResult(
                    source=str(pdf), status="rejected",
                    message=QUALITY_REJECTION_MESSAGE,
                    error_code="DOCUMENT_QUALITY_REJECTED", quality=quality,
                ))
                continue

        try:
            report = parse_pdf(pdf, work_dir / pdf.stem, company=lookup.get(pdf.name))
        except Exception as e:  # noqa: BLE001 —— 单个 PDF 失败不中断整批
            print(f"[FAIL] {pdf.name}: {e}")
            results.append(IngestionResult(
                source=str(pdf), status="failed", message=f"PDF 解析失败：{e}",
                error_code="PDF_PARSE_FAILED", quality=quality,
            ))
            continue
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        m = report["metainfo"]
        print(f"[OK] {pdf.name}: {m['num_pages']}页 {m['num_tables']}表 "
              f"{m['num_elements']}元素 {m['parse_seconds']}s")
        results.append(IngestionResult(
            source=str(pdf), status="accepted", message="文档解析成功",
            quality=quality, output_path=str(out),
        ))

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="解析年报 PDF → data/parsed/docs/*.json")
    ap.add_argument("pdfs", nargs="+", type=Path, help="PDF 文件或目录")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "parsed" / "docs")
    ap.add_argument("--no-quality-check", action="store_true",
                    help="跳过质量预检(不推荐)")
    args = ap.parse_args()

    paths: list[Path] = []
    for p in args.pdfs:
        paths.extend(sorted(p.glob("*.pdf")) if p.is_dir() else [p])
    parse_and_export(paths, args.out, quality_check=not args.no_quality_check)
