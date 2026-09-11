"""消融评测：表格序列化 vs 原始 HTML 的页级召回率（Recall@k）。

回答"表格语义化后相关段落召回率提升多少"：同一批题、同一个检索器，只有
表格块内容不同——
- A（serialized）：data/indexes + data/parsed/chunked（正常流水线，LLM 序列化表）
- B（raw_html）：data/indexes_raw_table + data/parsed/chunked_raw_table
  （text_splitter --table-source html 产物，MinerU 原始表格 HTML）

口径：
- 评测题：answers_eval.json 里带 reference_pools 的题（出处池 sha1:page），
  且池内文档已建索引；目标文档取自出处池（不走路由——这里测检索不测路由）
- Recall@k = |前 k 个候选 chunk 的 (doc, page) 对 ∩ 出处池| / |出处池对数|
  （多文档比较题：每个目标文档各检索 top-k 后合并计对）
- 生成端流程不变；本脚本只测检索融合层（top-20 候选，k=5/10/20 在其中取前缀）

页号基准：MinerU 页号与 gt pool 页号一致（2026-09-11 试跑验证——pool 页与
检索页字面相等命中，无 0/1-based 偏移）；如遇异常仍可用 --page-offset 校准。

用法：
    python -m enterprise_rag.eval_recall [--ks 5,10,20] [--top-n 20] \
        [--a-indexes DIR --a-chunked DIR] [--b-indexes DIR --b-chunked DIR] \
        [--page-offset 0] [--out data/eval_recall_result.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from enterprise_rag.retrieval.retriever import Retriever

# src/enterprise_rag/eval_recall.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _load_meta() -> dict[str, str]:
    """sha1 -> doc_id（csv 的 filename stem）。"""
    import csv

    out: dict[str, str] = {}
    with open(DATA_DIR / "pdf_metadata.csv", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["sha1"]] = Path(row["filename"]).stem
    return out


def _targets(meta: dict, gt: dict, indexes: set[str]) -> dict[str, list[list[tuple[str, int]]]]:
    """题目 -> [pool]，pool = [(doc_id, page)]。reference_pools 是"可接受出处
    池列表的列表"（41 题单池；8 题 5 池=比较题等多文档，每池归属一家；
    实测还有 pool 混入他家页的情况）。pool 内任一文档未建索引则整池剔除
    （部分目标无法算分），全空剔题。"""
    out: dict[str, list[list[tuple[str, int]]]] = {}
    for q, entry in gt.items():
        pools = []
        for pool in entry.get("reference_pools") or []:
            pairs = []
            for ref in pool:
                sha1, _, page = ref.rpartition(":")
                doc = meta.get(sha1, "")
                if doc not in indexes:
                    pairs = None
                    break
                pairs.append((doc, int(page)))
            if pairs:
                pools.append(pairs)
        if pools:
            out[q] = pools
    return out


def _recall_at(
    retriever: Retriever,
    question: str,
    pools: list[list[tuple[str, int]]],
    ks: list[int],
    top_n: int,
    page_offset: int,
) -> dict[int, float]:
    """每个池算 (doc, page) 对召回，题级 = 池均值（每家出处都要召回）。"""
    docs = {d for pool in pools for d, _ in pool}
    pages_by_doc: dict[str, list[int]] = {}
    for doc in docs:
        cands = retriever.retrieve(question, doc, top_n=top_n)
        seq: list[int] = []
        for c in cands:
            if c["page"] not in seq:
                seq.append(c["page"])
        pages_by_doc[doc] = seq
    out: dict[int, float] = {}
    for k in ks:
        per_pool = []
        for pool in pools:
            want = {(d, p + page_offset) for d, p in pool}
            got = {(d, p) for d, p in want if p in pages_by_doc.get(d, [])[:k]}
            per_pool.append(len(got) / len(want))
        out[k] = sum(per_pool) / len(per_pool)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="消融评测：表格序列化 vs 原始 HTML 召回率")
    ap.add_argument("--questions", type=Path, default=DATA_DIR / "questions_eval.json")
    ap.add_argument("--gt", type=Path, default=DATA_DIR / "answers_eval.json")
    ap.add_argument("--a-indexes", type=Path, default=DATA_DIR / "indexes")
    ap.add_argument("--a-chunked", type=Path, default=DATA_DIR / "parsed" / "chunked")
    ap.add_argument("--b-indexes", type=Path, default=DATA_DIR / "indexes_raw_table")
    ap.add_argument("--b-chunked", type=Path,
                    default=DATA_DIR / "parsed" / "chunked_raw_table")
    ap.add_argument("--ks", default="5,10,20", help="Recall@k 的 k 列表")
    ap.add_argument("--top-n", type=int, default=20, help="融合候选条数（k 的上限）")
    ap.add_argument("--page-offset", type=int, default=0,
                    help="gt 页号 -> MinerU 页号的偏移（试跑时校准）")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eval_recall_result.json")
    ap.add_argument("--show", type=int, default=3, help="打印每版前几题命中明细")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    meta = _load_meta()
    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    qset = {q["text"] for q in questions}

    # 目标文档必须两套索引都有，保证同题同文档对比
    a_indexed = {p.stem for p in args.a_indexes.glob("*.faiss")}
    b_indexed = {p.stem for p in args.b_indexes.glob("*.faiss")}
    indexed = a_indexed & b_indexed
    targets = _targets(meta, gt, indexed)
    targets = {q: v for q, v in targets.items() if q in qset}
    print(f"可评测题：{len(targets)}（两套索引均有 {len(indexed)} 份文档）")

    from enterprise_rag.indexing.ingestor import make_embedder
    emb = make_embedder()
    ret_a = Retriever(indexes_dir=args.a_indexes, chunked_dir=args.a_chunked,
                      embedder=emb)
    ret_b = Retriever(indexes_dir=args.b_indexes, chunked_dir=args.b_chunked, embedder=emb)

    rows = []
    for i, (q, pools) in enumerate(sorted(targets.items()), 1):
        ra = _recall_at(ret_a, q, pools, ks, args.top_n, args.page_offset)
        rb = _recall_at(ret_b, q, pools, ks, args.top_n, args.page_offset)
        rows.append({"q": q, "docs": sorted({d for pool in pools for d, _ in pool}),
                     "a": ra, "b": rb})
        if i <= args.show:

            def _fmt(r: dict[int, float]) -> str:
                return " ".join(f"@{k}={r[k]:.2f}" for k in ks)

            print(f"  [{i}] {q[:60]}")
            print(f"      A(serialized) {_fmt(ra)} | B(raw_html) {_fmt(rb)}")

    def agg(key: str) -> dict[int, float]:
        return {k: sum(r[key][k] for r in rows) / len(rows) for k in ks} if rows else {}

    a_avg, b_avg = agg("a"), agg("b")
    print(f"\n== 页级召回率（{len(rows)} 题均值） ==")
    print(f"{'指标':<12}{'A serialized':>14}{'B raw_html':>12}{'Δ(百分点)':>10}")
    for k in ks:
        d = (a_avg[k] - b_avg[k]) * 100
        print(f"Recall@{k:<7}{a_avg[k]:>14.3f}{b_avg[k]:>12.3f}{d:>+9.1f}")

    args.out.write_text(json.dumps(
        {"n": len(rows), "ks": ks,
         "a": {str(k): a_avg[k] for k in ks}, "b": {str(k): b_avg[k] for k in ks},
         "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细已存 {args.out}")


if __name__ == "__main__":
    main()
