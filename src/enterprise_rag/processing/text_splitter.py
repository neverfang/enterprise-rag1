"""文本切分：serialized/*.json -> chunked/*.json（小 chunk + 页父文本）。

对应原项目 src/text_splitter.py（RecursiveCharacterTextSplitter 300/50），两处适配：
- 切分策略更"语言化"：以**标题为硬边界**分段，段内贪心合并相邻段落到
  chunk_size（小段落单独成块会上下文太稀），只有单段超长时才降级按行/词硬切，
  断口处给 overlap——即"结构优先、大小封顶"，而非固定滑动窗口
- 每个正文 chunk 带标题面包屑前缀（如 "Financial Review > Dividend"），提高
  chunk 的上下文自含性（表格序列化思想的正文版，零 LLM 成本）

chunk 类型：
- content：正文打包块（含 equation 等杂项文本）
- serialized_table：一张已序列化表 = 一整块（information_blocks 拼接，不切碎）
- serialized_image：一张携带数据的图片的 vl 转写 = 一整块（3.5 步产物）

父文档：pages[] 为页级文本（标题/正文/序列化块/vl 转写按流序拼接），供检索
阶段小 chunk -> 大上下文回溯。PDF 分页是排版单位不是语义单位，为支持**跨页**
父窗口，每个 chunk 额外记录它在原内容流中的元素区间 els:[start, end)，
第 6 步可按元素窗口（而非页）构建父上下文。

用法：
    python -m enterprise_rag.processing.text_splitter [input_dir] \
        [--out DIR] [--chunk-size 300] [--overlap 50]

token 计数用 tiktoken o200k_base（与原版一致；对 DeepSeek 是近似，仅用于切块）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

from enterprise_rag.processing.normalize import normalize_text

# src/enterprise_rag/processing/text_splitter.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

_ENC = tiktoken.get_encoding("o200k_base")


def _ntok(text: str) -> int:
    return len(_ENC.encode(text)) if text else 0


class _HeadingStack:
    """维护当前标题层级栈，提供面包屑（"Financial Review > Dividend"）。"""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, text))

    def crumb(self) -> str:
        return " > ".join(t for _, t in self._stack)


def _split_long_text(text: str, budget: int, overlap: int) -> list[str]:
    """单个超长段落：先按行打包；仍超长的行按词滑窗，断口保留 ~overlap tokens。"""
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for line in text.split("\n"):
        t = _ntok(line)
        if t > budget:
            if cur:
                pieces.append("\n".join(cur))
                cur, cur_tok = [], 0
            pieces.extend(_split_line_by_words(line, budget, overlap))
            continue
        if cur and cur_tok + t > budget:
            pieces.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(line)
        cur_tok += t
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _split_line_by_words(line: str, budget: int, overlap: int) -> list[str]:
    words = line.split(" ")
    pieces: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for w in words:
        t = _ntok(w) + 1  # +1 粗略计入空格
        if cur and cur_tok + t > budget:
            pieces.append(" ".join(cur))
            # 尾部 ~overlap tokens 作为下一块开头（词对齐）
            tail: list[str] = []
            tail_tok = 0
            for tw in reversed(cur):
                tt = _ntok(tw) + 1
                if tail_tok + tt > overlap:
                    break
                tail.insert(0, tw)
                tail_tok += tt
            cur, cur_tok = tail, tail_tok
        cur.append(w)
        cur_tok += t
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def _pack_prose(atoms: list[dict], budget: int, overlap: int) -> list[list[dict]]:
    """段内贪心打包：{text, toks, idx, page} 原子 -> 原子组列表。

    相邻段落合并到 <=budget；超长段落单独硬切（其碎片不与邻段再合并，保持断口清晰）。
    """
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_tok = 0
    for a in atoms:
        if a["toks"] > budget:
            if cur:
                groups.append(cur)
                cur, cur_tok = [], 0
            # 超长段的每个碎片各自成块（碎片间已有 overlap 衔接）
            for p in _split_long_text(a["text"], budget, overlap):
                groups.append([{**a, "text": p, "toks": _ntok(p)}])
            continue
        if cur and cur_tok + a["toks"] > budget:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(a)
        cur_tok += a["toks"]
    if cur:
        groups.append(cur)
    return groups


def split_doc(
    doc: dict,
    chunk_size: int = 300,
    overlap: int = 50,
    *,
    table_source: str = "serialized",
) -> dict:
    """一份 serialized 文档 -> {metainfo, pages, chunks}。

    table_source：表格块内容来源——
    - "serialized"（默认）：LLM 序列化的 information_blocks（正常流水线）
    - "html"：MinerU 原始表格 HTML（消融对照：表格未语义化，其余不变）
    """
    els = doc.get("content", [])
    tables = {t["id"]: t for t in doc.get("tables", [])}
    stack = _HeadingStack()

    # —— 第一遍：把内容流转成原子（prose 可打包；table/picture 各自成块）——
    prose_atoms: list[dict] = []   # 打包用
    unit_chunks: list[dict] = []   # table / picture 整块，带 idx 便于流序合并
    page_parts: dict[int, list[str]] = {}
    for idx, e in enumerate(els):
        etype = e.get("type")
        text = normalize_text(e.get("text") or "")
        page = e.get("page")
        if etype == "heading":
            stack.push(e.get("level") or 1, text)
            if text:
                page_parts.setdefault(page, []).append(
                    "#" * min(e.get("level") or 1, 4) + " " + text)
            continue
        if etype == "table":
            t = tables.get(e.get("table_id") or "")
            if table_source == "html":
                # 消融对照：表格保持原始 HTML（未语义化）；无 html 才兜底 caption
                body = ((f"Table: {t['caption']}\n" if (t or {}).get("caption") else "")
                        + ((t or {}).get("html") or text or "[table]"))
            else:
                ser = (t or {}).get("serialized")
                if ser:
                    blocks = "\n".join(b["information_block"]
                                       for b in ser["information_blocks"])
                    body = ((f"Table: {t['caption']}\n" if t.get("caption") else "")
                            + blocks)
                else:  # 未序列化的表（dev 集为 0）：caption 兜底，防整表信息丢失
                    body = text or "[table]"
                    print(f"  [WARN] 表 {e.get('table_id')} 无 serialized，仅 caption 入块")
            unit_chunks.append({
                "type": "serialized_table", "page": page, "text": body,
                "table_id": e.get("table_id"), "els": [idx, idx + 1],
            })
            page_parts.setdefault(page, []).append(body)
            continue
        if etype == "picture":
            vl = e.get("vl") or {}
            body = ""
            if text:  # caption
                body = text
            if vl.get("data") == "HAS_DATA" and vl.get("text"):
                body = (body + "\n" + vl["text"]) if body else vl["text"]
                unit_chunks.append({
                    "type": "serialized_image", "page": page, "text": vl["text"],
                    "img": vl.get("img"), "els": [idx, idx + 1],
                })
            if body:
                page_parts.setdefault(page, []).append(body)
            continue
        # paragraph / equation / 其他文本
        if text:
            prose_atoms.append({
                "text": text, "toks": _ntok(text), "idx": idx,
                "page": page, "crumb": stack.crumb(),
            })
            page_parts.setdefault(page, []).append(text)

    # —— 第二遍：按标题边界分段（面包屑变化处切开），段内打包 ——
    groups: list[list[dict]] = []
    for a in prose_atoms:
        if groups and groups[-1] and groups[-1][-1]["crumb"] == a["crumb"]:
            target = groups[-1]
        else:
            groups.append([])
            target = groups[-1]
        target.append(a)
    # 先按段分组再打包（标题是硬边界；段打包在段内做）
    packed: list[dict] = []
    for sec in groups:
        crumb = sec[0]["crumb"]
        prefix = (crumb + "\n\n") if crumb else ""
        budget = max(chunk_size - _ntok(prefix), 64)
        for g in _pack_prose(sec, budget, overlap):
            packed.append({
                "type": "content",
                "page": g[0]["page"],
                "text": prefix + "\n\n".join(a["text"] for a in g),
                "els": [g[0]["idx"], g[-1]["idx"] + 1],
            })

    # —— 合并两类块并按流序（起始元素）排序、编号 ——
    all_chunks = packed + unit_chunks
    all_chunks.sort(key=lambda c: c["els"][0])
    for n, c in enumerate(all_chunks):
        c["id"] = n
        c["length_tokens"] = _ntok(c["text"])

    pages = [{"page": p, "text": "\n\n".join(parts)}
             for p, parts in sorted(page_parts.items(), key=lambda kv: kv[0] or 0)]
    return {"metainfo": doc.get("metainfo", {}), "pages": pages, "chunks": all_chunks}


def split_all(
    input_dir: Path,
    out_dir: Path,
    chunk_size: int,
    overlap: int,
    *,
    transform: bool = False,
    client=None,
    model: str | None = None,
    transform_workers: int = 4,
    table_source: str = "serialized",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if transform and (client is None or model is None):
        from enterprise_rag.processing.chunk_transformer import make_client

        client, model = make_client()
    for in_path in sorted(input_dir.glob("*.json")):
        doc = json.loads(in_path.read_text(encoding="utf-8"))
        result = split_doc(doc, chunk_size, overlap, table_source=table_source)
        if transform:
            from enterprise_rag.processing.chunk_transformer import transform_document

            result = transform_document(
                result,
                client=client,
                model=model,
                max_workers=transform_workers,
            )
        (out_dir / in_path.name).write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        from collections import Counter
        tc = Counter(c["type"] for c in result["chunks"])
        toks = sorted(c["length_tokens"] for c in result["chunks"])
        n = len(toks)
        p50 = toks[n // 2] if n else 0
        p95 = toks[int(n * 0.95)] if n else 0
        print(f"[OK] {in_path.name}: {len(result['pages'])} 页，"
              f"chunks {dict(tc)}（共 {n}），"
              f"tokens p50={p50}/p95={p95}/max={toks[-1] if toks else 0}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第 4 步：文本切分（serialized -> chunked，小 chunk + 页父文本）")
    ap.add_argument("input", type=Path, nargs="?", default=DATA_DIR / "parsed" / "serialized",
                    help="输入目录（默认 data/parsed/serialized）")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "parsed" / "chunked")
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument(
        "--transform", action="store_true",
        help="显式启用正文 Chunk LLM 清洗（默认关闭，会产生 API 费用）",
    )
    ap.add_argument("--transform-workers", type=int, default=4)
    ap.add_argument(
        "--table-source", choices=["serialized", "html"], default="serialized",
        help="表格块内容：serialized=LLM 序列化信息块（默认）；html=原始表格 HTML"
             "（消融对照，产出可配合 --out data/parsed/chunked_raw_table）",
    )
    args = ap.parse_args()
    split_all(
        args.input,
        args.out,
        args.chunk_size,
        args.overlap,
        transform=args.transform,
        transform_workers=args.transform_workers,
        table_source=args.table_source,
    )


if __name__ == "__main__":
    main()
