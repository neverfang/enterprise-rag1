"""检索：问题 -> 公司路由 -> 向量+BM25 双路召回 -> 融合候选列表（第 7 步重排的输入）。

对应原项目 retrieval.py，结构对照（2026-09-08 讨论）：
- 路由：原版 _extract_companies_from_subset 的正则法——遍历公司名在问题里做
  词边界匹配，命中即从文本删除继续找（天然支持比较题多公司）；我们匹配键 =
  公司名 ∪ 文件名 stem（dev 题会用 "TSX_Y" 这种 stem 指代公司）
- 召回：原版只走向量 top-28（BM25Retriever 建了没用）；我们升级为双路（方案 B），
  依据是 dev 5 题抽查两路 top3 重叠仅 0-1/3：向量补同义词（buyback→repurchase）、
  BM25 补精确词/数字。融合 = 各路分数在自身 top-k 内 min-max 归一化后加权，
  缺失路记 0（偏向两路都命中的块；只命中一路的仍能进候选池交给重排）
- 父文档回溯：独立接口 parent_context()，供第 7 步重排后调用。两种模式：
  - pages（同原版，pipeline 默认）：chunk -> 所在页整页文本，按页去重
  - els_window：用 chunk 的 els 元素区间向两侧扩 window 个元素、从内容流拼文本，
    相邻区间合并去重——跨页的语义连续段落不被页边界切断（第 4 步预留的钩子）。
    总字符超 max_chars（默认 60K，对齐生成端保险丝）时窗口减半重建而非丢段，
    保住所有候选的邻域；减到 window=0 仍超（极端：多个超大表块）由调用方兜底

用法（冒烟）：
    python -m enterprise_rag.retrieval.retriever "问题" \
        [--k 30] [--vw 0.6] [--top-n 20] [--parent pages|els_window] [--window 15]
        [--max-chars 60000]
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np

from enterprise_rag.indexing.ingestor import bm25_tokenize, make_embedder
from enterprise_rag.processing.normalize import normalize_text

# src/enterprise_rag/retrieval/retriever.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


class Router:
    """问题 -> doc_id 列表（正则词边界匹配公司名/stem，忠实原版做法）。"""

    def __init__(self, meta_csv: Path, chunked_dir: Path) -> None:
        entries = []
        with open(meta_csv, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                stem = Path(row["filename"]).stem
                if not (chunked_dir / f"{stem}.json").exists():
                    continue  # 只路由已建索引的文档
                entries.append({"doc_id": stem, "keys": [row["company"], stem]})
        if not entries:
            raise SystemExit(f"路由表为空：{meta_csv} 与 {chunked_dir} 无交集")
        # 长键优先（避免短名先命中吞掉长名的部分）
        self._patterns = sorted(
            ((k, e["doc_id"]) for e in entries for k in e["keys"] if k),
            key=lambda x: -len(x[0]))

    def route(self, question: str) -> list[str]:
        rest = question
        hits: list[str] = []
        for key, doc_id in self._patterns:
            m = re.search(re.escape(key) + r"(?:\W|$)", rest, re.IGNORECASE)
            if m:
                hits.append(doc_id)
                rest = rest[:m.start()] + " " + rest[m.end():]
        # 去重保序（比较题多家/同文档多键命中）
        seen: list[str] = []
        for d in hits:
            if d not in seen:
                seen.append(d)
        return seen


class Retriever:
    """单文档检索器：双路召回 + 融合 + 父文档回溯。文档级缓存，查询端复用 Embedder。"""

    def __init__(self, indexes_dir: Path | None = None,
                 chunked_dir: Path | None = None,
                 serialized_dir: Path | None = None, embedder=None) -> None:
        self.indexes_dir = indexes_dir or DATA_DIR / "indexes"
        self.chunked_dir = chunked_dir or DATA_DIR / "parsed" / "chunked"
        self.serialized_dir = serialized_dir or DATA_DIR / "parsed" / "serialized"
        self.embedder = embedder or make_embedder()
        self._docs: dict[str, dict] = {}   # doc_id -> chunked 文档 + 双路索引
        self._ser: dict[str, dict] = {}    # doc_id -> serialized 文档（仅 els_window 用）

    def _doc(self, doc_id: str) -> dict:
        if doc_id not in self._docs:
            chunked = json.loads(
                (self.chunked_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
            index = faiss.read_index(str(self.indexes_dir / f"{doc_id}.faiss"))
            with open(self.indexes_dir / f"{doc_id}.pkl", "rb") as f:
                bm = pickle.load(f)
            self._docs[doc_id] = {
                "chunks": {c["id"]: c for c in chunked["chunks"]},
                "pages": {p["page"]: p for p in chunked["pages"]},
                "index": index, "bm25": bm["bm25"],
            }
        return self._docs[doc_id]

    def _serialized(self, doc_id: str) -> dict:
        if doc_id not in self._ser:
            self._ser[doc_id] = json.loads(
                (self.serialized_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        return self._ser[doc_id]

    def retrieve(self, question: str, doc_id: str, k: int = 30,
                 vector_weight: float = 0.6, top_n: int = 20) -> list[dict]:
        """双路召回 + 融合，返回按融合分排序的候选 chunk 列表（top_n 个）。"""
        d = self._doc(doc_id)
        n = len(d["chunks"])
        k = min(k, n)

        # 向量路（IndexFlatIP 内积 = 余弦，越大越好）
        qv = self.embedder.embed_query(question).reshape(1, -1)
        D, I = d["index"].search(qv, k)
        vec = {int(i): float(s) for s, i in zip(D[0], I[0])}

        # BM25 路
        scores = d["bm25"].get_scores(bm25_tokenize(question))
        bm = {i: float(scores[i]) for i in sorted(
            range(n), key=lambda j: -scores[j])[:k]}

        def _norm(sc: dict[int, float]) -> dict[int, float]:
            if not sc:
                return {}
            lo, hi = min(sc.values()), max(sc.values())
            rng = hi - lo
            return {i: (v - lo) / rng if rng > 1e-9 else 1.0 for i, v in sc.items()} \
                if rng > 1e-9 else {i: 1.0 for i in sc}

        nvec, nbm = _norm(vec), _norm(bm)
        union = set(vec) | set(bm)
        fused = sorted(
            union,
            key=lambda cid: -(vector_weight * nvec.get(cid, 0.0)
                              + (1 - vector_weight) * nbm.get(cid, 0.0)))[:top_n]

        out = []
        for cid in fused:
            c = d["chunks"][cid]
            out.append({
                "chunk_id": cid, "type": c["type"], "page": c["page"],
                "els": c.get("els"), "text": c["text"],
                "score_vec": round(nvec.get(cid, 0.0), 4),
                "score_bm25": round(nbm.get(cid, 0.0), 4),
                "score": round(vector_weight * nvec.get(cid, 0.0)
                               + (1 - vector_weight) * nbm.get(cid, 0.0), 4),
            })
        return out

    # ---------- 父文档回溯 ----------

    def parent_context(self, doc_id: str, candidates: list[dict],
                       mode: str = "pages", window: int = 15,
                       max_chars: int = 60_000) -> list[dict]:
        """候选 chunks -> 父文本列表（去重保序）。第 7 步重排后对 top 候选调用。

        - pages：chunk 所在页整页文本，按页去重（同原版 return_parent_pages）
        - els_window：els 区间向两侧扩 window 个元素、内容流拼文本，
          相邻/重叠区间合并（跨页语义连续，第 4 步钩子）；总字符超 max_chars
          时 window 减半重建（全部候选邻域都在，只是变窄），减到 0 仍超
          （多个超大表块挤进 top）则原样返回、由调用方保险丝兜底
        """
        d = self._doc(doc_id)
        if mode == "pages":
            seen: set = set()
            out = []
            for c in candidates:
                p = c["page"]
                if p in seen:
                    continue
                seen.add(p)
                out.append({"page": p, "text": d["pages"][p]["text"]})
            return out
        if mode == "els_window":

            def build(w: int) -> list[list[int]]:
                spans = []
                for c in candidates:
                    s, e = c["els"]
                    spans.append([max(0, s - w), e + w])
                spans.sort()
                merged: list[list[int]] = []
                for s, e in spans:  # 合并重叠/相邻区间
                    if merged and s <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([s, e])
                return merged

            w = window
            merged = build(w)
            texts = [self._elements_text(doc_id, s, e) for s, e in merged]
            while w > 0 and sum(len(t) for t in texts) > max_chars:
                w //= 2
                merged = build(w)
                texts = [self._elements_text(doc_id, s, e) for s, e in merged]
            return [{"els": [s, e], "text": t} for (s, e), t in zip(merged, texts)]
        raise ValueError(f"未知父文档模式: {mode}")

    def _elements_text(self, doc_id: str, lo: int, hi: int) -> str:
        """内容流元素区间 -> 文本（与 text_splitter 页组装同规则：标题 #、表序列化块、vl 转写）。"""
        doc = self._serialized(doc_id)
        tables = {t["id"]: t for t in doc.get("tables", [])}
        parts: list[str] = []
        for e in doc["content"][lo:hi]:
            text = normalize_text(e.get("text") or "")
            etype = e.get("type")
            if etype == "heading":
                if text:
                    parts.append("#" * min(e.get("level") or 1, 4) + " " + text)
            elif etype == "table":
                ser = (tables.get(e.get("table_id") or "") or {}).get("serialized")
                if ser:
                    parts.append("\n".join(b["information_block"]
                                           for b in ser["information_blocks"]))
                else:
                    parts.append(text or "[table]")
            elif etype == "picture":
                vl = e.get("vl") or {}
                if vl.get("data") == "HAS_DATA" and vl.get("text"):
                    parts.append(vl["text"])
                elif text:
                    parts.append(text)
            elif text:
                parts.append(text)
        return "\n\n".join(parts)


def _snippet(text: str, n: int = 90) -> str:
    return text[:n].replace("\n", " ") + ("…" if len(text) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第 6 步检索冒烟：路由 + 向量/BM25 双路 + 融合 + 父文档回溯")
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=30, help="每路召回条数")
    ap.add_argument("--vw", type=float, default=0.6, help="向量路融合权重")
    ap.add_argument("--top-n", type=int, default=20, help="融合候选输出条数")
    ap.add_argument("--parent", choices=["pages", "els_window"], default="pages")
    ap.add_argument("--window", type=int, default=15, help="els_window 两侧扩展元素数")
    ap.add_argument("--max-chars", type=int, default=60_000,
                    help="els_window 总字符预算（超出则窗口减半重建）")
    ap.add_argument("--show", type=int, default=10, help="打印融合 top 几")
    args = ap.parse_args()

    router = Router(DATA_DIR / "pdf_metadata.csv", DATA_DIR / "parsed" / "chunked")
    doc_ids = router.route(args.question)
    if not doc_ids:
        print("路由失败：问题里没匹配到任何已索引文档的公司名/stem", flush=True)
        sys.exit(1)
    note = "（比较题多公司：当前取第一家演示，改写与汇总在第 8 步）" \
        if len(doc_ids) > 1 else ""
    print(f"路由 -> {doc_ids}{note}", flush=True)

    r = Retriever(embedder=make_embedder())
    doc_id = doc_ids[0]
    cands = r.retrieve(args.question, doc_id, k=args.k,
                       vector_weight=args.vw, top_n=args.top_n)

    d = r._doc(doc_id)
    n = len(d["chunks"])
    print(f"\n== 双路各自 top5（共 {n} chunks） ==", flush=True)
    vec_top = sorted(cands, key=lambda c: -c["score_vec"])[:5]
    bm_top = sorted(cands, key=lambda c: -c["score_bm25"])[:5]
    for tag, rows in (("FAISS", vec_top), ("BM25 ", bm_top)):
        for c in rows:
            print(f" {tag} v={c['score_vec']:.2f} b={c['score_bm25']:.2f} "
                  f"[{c['type']}:p{c['page']}] {_snippet(c['text'])}", flush=True)

    print(f"\n== 融合 top{min(args.show, len(cands))}（vw={args.vw}） ==", flush=True)
    for i, c in enumerate(cands[:args.show], 1):
        print(f" {i:2d}. s={c['score']:.3f} v={c['score_vec']:.2f} b={c['score_bm25']:.2f} "
              f"[{c['type']}:p{c['page']}] {_snippet(c['text'])}", flush=True)

    parents = r.parent_context(doc_id, cands[:6], mode=args.parent,
                               window=args.window, max_chars=args.max_chars)
    total = sum(len(p["text"]) for p in parents)
    where = [f"p{p['page']}" for p in parents] if args.parent == "pages" \
        else [f"els{p['els']}({len(p['text'])})" for p in parents]
    budget = f"，预算 {args.max_chars}" if args.parent == "els_window" else ""
    print(f"\n== 父文档（top6 chunks -> {args.parent}，{len(parents)} 段共 "
          f"{total} 字符{budget}）: {' '.join(where)} ==", flush=True)
    if parents:
        print(_snippet(parents[0]["text"], 200), flush=True)


if __name__ == "__main__":
    main()
