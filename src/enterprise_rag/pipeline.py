"""端到端批量问答：题目 json -> 路由->检索->重排->父文档->生成 -> answers_run.json。

第 9 步的串联层：不新增检索/生成逻辑，复用 Router / Retriever / Reranker /
Generator。对应原项目 questions_processing.py 的 process_questions_list 职责。

忠实原版的比较题三步（process_comparative_question，提示词自写简版）：
1. LLM 把比较题改写成每家公司的独立子问题
2. 各家并行作答（子问题一律 number 型——原版即如此硬编码，比较题本质是取数）
3. LLM 汇总各家答案回答原题

我们的工程处理：
- 题目字段自适应（text|question、kind|schema），兼容 test_set / round2 / eval
- 断点续跑：输出文件里已答的题（按题目文本对齐）跳过，每答完一题即落盘
- 路由 0 家 -> 直接 N/A（eval 100 份全建索引后不应发生，dev 部分建索引时
  用来跳过未索引公司）
- references 输出 [{pdf_sha1, page_index}]（sha1 来自 pdf_metadata.csv），
  评测引用有效性用

用法：
    python -m enterprise_rag.pipeline <questions.json> \
        [--out data/answers_run.json] [--limit N] [--workers 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from enterprise_rag.generation.generator import Generator
from enterprise_rag.reranking.reranker import Reranker, make_client
from enterprise_rag.retrieval.retriever import DATA_DIR, Router, Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# —— 比较题提示词（自写简版，流程忠实原版三步）——
REPHRASE_SYSTEM = """You rephrase a comparative question about multiple companies into one standalone question per company.
Each sub-question must be self-contained (name the company explicitly), ask for the specific metric or fact from the original question, and keep the original wording style.
Return only valid json (no markdown fences): {"<company>": "<sub-question>", ...}
"""

COMBINE_SYSTEM = """You are given answers to per-company sub-questions extracted from annual reports. Combine them to answer the original comparative question.
Return only valid json (no markdown fences) of exactly this shape:
{
  "final_answer": "...",
  "reasoning_summary": "..."
}
final_answer rules:
- Directly answer the original question (usually a company name, sometimes a number or True/False).
- Extracted exactly as it appears in the sub-answers; no extra words or comments.
- Return 'N/A' if the needed values are not available.
"""


class Pipeline:
    def __init__(self, top_n: int = 6, weak_th: float = 0.5) -> None:
        self.router = Router(DATA_DIR / "pdf_metadata.csv", DATA_DIR / "parsed" / "chunked")
        from enterprise_rag.indexing.ingestor import make_embedder
        self.retriever = Retriever(embedder=make_embedder())
        client, model = make_client()          # 重排/生成共享一个客户端
        self.reranker = Reranker(client=client, model=model)
        self.generator = Generator(client=client, model=model)
        self.top_n = top_n
        self.weak_th = weak_th
        # doc_id -> sha1（references 用；csv 里 filename 相对 data/raw/）
        self._sha1: dict[str, str] = {}
        with open(DATA_DIR / "pdf_metadata.csv", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                self._sha1[Path(row["filename"]).stem] = row["sha1"]

    # ---------- 单公司全链路 ----------

    def _single(self, doc_id: str, question: str, kind: str) -> dict:
        cands = self.retriever.retrieve(question, doc_id, top_n=20)
        top = self.reranker.rerank(question, cands, top_n=self.top_n)
        parents = self.retriever.parent_context(doc_id, top)
        kept, total = [], 0
        for p in parents:                      # 60K 字符保险丝（同 generator CLI）
            if total + len(p["text"]) > 60000:
                break
            kept.append(p)
            total += len(p["text"])
        top_llm = max(c["relevance_score"] for c in top) if top else 0.0
        ans = self.generator.answer(
            question, kept, kind=kind,
            weak_evidence=self.weak_th > 0 and top_llm <= self.weak_th)
        ans["references"] = [
            {"pdf_sha1": self._sha1.get(doc_id, ""), "page_index": p}
            for p in ans.get("relevant_pages", [])]
        return ans

    # ---------- 比较题三步 ----------

    def _comparative(self, question: str, doc_ids: list[str]) -> dict:
        # 1) 改写成每家子问题
        resp = self.generator.client.chat.completions.create(
            model=self.generator.model, temperature=0, max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REPHRASE_SYSTEM},
                {"role": "user",
                 "content": f'Original question: "{question}"\n\nCompanies: {doc_ids}'},
            ], timeout=120)
        sub_qs: dict[str, str] = json.loads(resp.choices[0].message.content or "{}")
        # 2) 各家独立作答（number 型，同原版）；路由键=doc_id，子问题键对不上时按序兜底
        sub_answers: dict[str, dict] = {}
        for i, doc_id in enumerate(doc_ids):
            sq = sub_qs.get(doc_id) or (
                sorted(sub_qs.values())[i] if i < len(sub_qs)
                else f"For {doc_id}, " + question)
            sub_answers[doc_id] = self._single(doc_id, sq, "number")
        # 3) 汇总
        ctx = json.dumps(
            {d: {"final_answer": a["final_answer"],
                 "reasoning_summary": a.get("reasoning_summary", ""),
                 "pages": a.get("relevant_pages", [])}
             for d, a in sub_answers.items()}, ensure_ascii=False)
        resp = self.generator.client.chat.completions.create(
            model=self.generator.model, temperature=0, max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": COMBINE_SYSTEM},
                {"role": "user",
                 "content": f'Original question: "{question}"\n\nSub-answers:\n{ctx}'},
            ], timeout=120)
        combined = json.loads(resp.choices[0].message.content or "{}")
        return {
            "final_answer": combined.get("final_answer", "N/A"),
            "reasoning_summary": combined.get("reasoning_summary", ""),
            "relevant_pages": [],               # 跨公司页码无比较意义，见每家子答案
            "sub_answers": {d: a["final_answer"] for d, a in sub_answers.items()},
            "references": [
                {"pdf_sha1": self._sha1.get(d, ""), "page_index": p}
                for d, a in sub_answers.items()
                for p in a.get("relevant_pages", [])],
        }

    # ---------- 题目入口 ----------

    def answer_question(self, question: str, kind: str) -> dict:
        doc_ids = self.router.route(question)
        if not doc_ids:
            return {"final_answer": "N/A",
                    "reasoning_summary": "路由失败：未匹配到已建索引的公司",
                    "relevant_pages": [], "references": []}
        if len(doc_ids) == 1:
            return self._single(doc_ids[0], question, kind)
        return self._comparative(question, doc_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="第 9 步：批量端到端问答")
    ap.add_argument("questions", type=Path)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "answers_run.json")
    ap.add_argument("--limit", type=int, help="最多处理 N 题（冒烟用）")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    raw = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [
        {"question": q.get("text") or q.get("question"),
         "kind": q.get("kind") or q.get("schema") or "name"}
        for q in raw]
    if args.limit is not None:
        questions = questions[:args.limit]

    done: dict[str, dict] = {}                  # 题目文本 -> 已有答案（断点续跑）
    if args.out.exists():
        for a in json.loads(args.out.read_text(encoding="utf-8")):
            if "error" in a:                    # 失败行不算已答，续跑时重试
                continue
            done[a["question"]] = a
        print(f"# 续跑：{args.out} 已有 {len(done)} 题", flush=True)

    pipe = Pipeline()
    todo = [q for q in questions if q["question"] not in done]
    print(f"# 待答 {len(todo)} / 共 {len(questions)} 题，"
          f"workers={args.workers}", flush=True)

    def run_one(q: dict) -> dict:
        try:
            ans = pipe.answer_question(q["question"], q["kind"])
        except Exception as e:  # noqa: BLE001 —— 单题失败不中断整批，标 error 续跑可重试
            ans = {"final_answer": "N/A", "reasoning_summary": f"[ERROR] {e}",
                   "relevant_pages": [], "references": [], "error": str(e)}
        doc_ids = pipe.router.route(q["question"])
        return {"question": q["question"], "kind": q["kind"], "doc_ids": doc_ids,
                **ans}

    n_na = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, q): q for q in todo}
        for fut in as_completed(futs):
            row = fut.result()
            done[row["question"]] = row
            if "error" in row:
                n_err += 1
            if str(row["final_answer"]) == "N/A":
                n_na += 1
            args.out.write_text(
                json.dumps(list(done.values()), ensure_ascii=False, indent=1),
                encoding="utf-8")               # 每题落盘，中断可续
            print(f"[{len(done)}/{len(questions)}] {row['kind']:7s} "
                  f"{str(row['final_answer'])[:60]!r} <- {row['doc_ids']}",
                  flush=True)

    print(f"# 完成：{len(done)} 题（本轮 N/A {n_na}、错误 {n_err}）-> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
