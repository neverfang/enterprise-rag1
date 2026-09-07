"""LLM 重排序：第 6 步融合 top-20 候选 -> DeepSeek 逐块相关性打分 -> combined 排序取 top-N。

对应原项目 reranking.py 的 LLMReranker，结构对照（2026-09-08 讨论）：
- 打分 prompt：忠实沿用原版的 0-1 锚点量表（0.1 步进、每档文字定义），
  多块格式 Block N + 分隔线，要求"exactly N rankings, in order"
- combined = llm_weight×LLM 分 + (1-llm_weight)×检索分（默认 0.7/0.3）。
  原版加权项是向量路的 distance；我们换成第 6 步的融合分（0-1、越大越好、
  已含双路信息）——语义方向一致且信息更全
- 数量不匹配兜底：重试 3 次仍缺分才补 0.0（同原版，但我们先重试再兜底）

超出原版的调整：
- 批量默认 10 块/prompt（原版调用处 2 块 -> 20 候选要发 10 个请求）；
  DeepSeek json 模式一次可稳定给多块打分，20 候选 = 2 个请求。批内互相
  干扰的风险用冒烟对比验证（--batch 2 切回原版粒度）
- 每块文本截断 cap（默认 1,500 o200k tok）：表块最大 5,700+ tok，打相关性
  分用不到全文；只影响重排输入，第 8 步生成仍用完整块/父文档
- 模型 deepseek-chat（temperature=0，json_object 模式 + pydantic 校验，
  与 table_serializer 同一套调用模式）

用法（冒烟，串联第 6 步）：
    python -m enterprise_rag.reranking.reranker "问题" \
        [--batch 10] [--llm-w 0.7] [--top-n 6] [--cap 1500] [--show 6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from enterprise_rag.retrieval.retriever import DATA_DIR, Router, Retriever, _snippet

_ENC = tiktoken.get_encoding("o200k_base")

# —— 忠实沿用原版的 0-1 锚点量表（prompts.RerankingPrompt），仅格式段自写 ——
SYSTEM_PROMPT = """You are a RAG (Retrieval-Augmented Generation) retrievals ranker.

You will receive a query and several retrieved text blocks related to that query. Your task is to evaluate and score each block based on its relevance to the query provided.

Instructions:

1. Reasoning:
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context relevant to the query. Explain your reasoning in a few sentences, referencing specific elements of the block to justify your evaluation. Avoid assumptions—focus solely on the content provided.

2. Relevance Score (0 to 1, in increments of 0.1):
   0 = Completely Irrelevant: The block has no connection or relation to the query.
   0.1 = Virtually Irrelevant: Only a very slight or vague connection to the query.
   0.2 = Very Slightly Relevant: Contains an extremely minimal or tangential connection.
   0.3 = Slightly Relevant: Addresses a very small aspect of the query but lacks substantive detail.
   0.4 = Somewhat Relevant: Contains partial information that is somewhat related but not comprehensive.
   0.5 = Moderately Relevant: Addresses the query but with limited or partial relevance.
   0.6 = Fairly Relevant: Provides relevant information, though lacking depth or specificity.
   0.7 = Relevant: Clearly relates to the query, offering substantive but not fully comprehensive information.
   0.8 = Very Relevant: Strongly relates to the query and provides significant information.
   0.9 = Highly Relevant: Almost completely answers the query with detailed and specific information.
   1 = Perfectly Relevant: Directly and comprehensively answers the query with all the necessary specific information.

3. Additional Guidance:
   - Objectivity: Evaluate blocks based only on their content relative to the query.
   - Clarity: Be clear and concise in your justifications.
   - No assumptions: Do not infer information beyond what's explicitly stated in the block.
"""

JSON_FORMAT_INSTRUCTION = """
Return only valid json (no markdown fences) of exactly this shape:
{
  "block_rankings": [
    {"reasoning": "...", "relevance_score": 0.0},
    ...
  ]
}
- block_rankings: exactly one entry per input block, in the same order.
- relevance_score: 0 to 1 in increments of 0.1 (see the score anchors above).
"""


class BlockRanking(BaseModel):
    reasoning: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class BatchRankings(BaseModel):
    block_rankings: list[BlockRanking]


def make_client() -> tuple[OpenAI, str]:
    """从 .env 读取 LLM 配置（与 table_serializer 同源），返回 (客户端, 模型名)。"""
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key or "填" in api_key:
        raise SystemExit("请先在项目根目录 .env 里配置 DEEPSEEK_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    # trust_env=False / max_retries=0：理由同 ingestor（国内端点直连、
    # SDK 重试与自定义重试叠加会拖长最坏等待）
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0,
                    http_client=httpx.Client(trust_env=False))
    return client, model


class Reranker:
    """融合候选列表 -> LLM 相关性打分 -> combined 排序。"""

    def __init__(self, client: OpenAI | None = None, model: str | None = None,
                 llm_weight: float = 0.7, batch: int = 10, cap: int = 1500,
                 retries: int = 3) -> None:
        if client is None or model is None:
            client, model = make_client()
        self.client = client
        self.model = model
        self.llm_weight = llm_weight
        self.batch = batch
        self.cap = cap            # 每块文本截断（o200k tok），仅重排输入
        self.retries = retries
        self.tokens_in = 0
        self.tokens_out = 0

    # ---------- 内部 ----------

    def _truncate(self, text: str) -> str:
        ids = _ENC.encode(text)
        return text if len(ids) <= self.cap else _ENC.decode(ids[:self.cap])

    def _score_batch(self, question: str, batch: list[dict]) -> list[BlockRanking]:
        """一批候选 -> 与输入等长等序的打分列表（失败重试，耗尽兜底 0 分）。"""
        formatted = "\n\n---\n\n".join(
            f'Block {i + 1}:\n\n"""\n{self._truncate(c["text"])}\n"""'
            for i, c in enumerate(batch))
        user_prompt = (
            f'Here is the query: "{question}"\n\n'
            f"Here are the retrieved text blocks:\n{formatted}\n\n"
            f"You should provide exactly {len(batch)} rankings, in order."
        )
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=4096,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system",
                         "content": SYSTEM_PROMPT + JSON_FORMAT_INSTRUCTION},
                        {"role": "user", "content": user_prompt},
                    ], timeout=120)
                usage = resp.usage
                if usage:
                    self.tokens_in += usage.prompt_tokens
                    self.tokens_out += usage.completion_tokens
                parsed = BatchRankings.model_validate(
                    json.loads(resp.choices[0].message.content or ""))
                if len(parsed.block_rankings) != len(batch):
                    raise ValueError(
                        f"返回 {len(parsed.block_rankings)} 个打分 != 批大小 {len(batch)}")
                return parsed.block_rankings
            except Exception as e:  # noqa: BLE001 —— JSON 越界/数量不符/网络，都走重试
                last_err = e
                time.sleep(2 * (attempt + 1))
        print(f"  [WARN] 一批重排重试 {self.retries} 次仍失败（{last_err}），"
              f"{len(batch)} 块按 0 分兜底", flush=True)
        return [BlockRanking(reasoning="default: LLM 未返回该块打分",
                             relevance_score=0.0) for _ in batch]

    # ---------- 对外 ----------

    def rerank(self, question: str, candidates: list[dict], top_n: int = 6,
               workers: int = 2) -> list[dict]:
        """融合候选 -> 打分 -> combined 排序，返回前 top_n 个（附打分字段）。"""
        if not candidates:
            return []
        batches = [candidates[i:i + self.batch]
                   for i in range(0, len(candidates), self.batch)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ranked = list(ex.map(lambda b: self._score_batch(question, b), batches))
        # 兜底 0 分后排序自然沉底，不影响已成功打分的块
        out = []
        for c, rank in zip(candidates, (r for batch in ranked for r in batch)):
            d = dict(c)
            d["relevance_score"] = rank.relevance_score
            d["combined_score"] = round(
                self.llm_weight * rank.relevance_score
                + (1 - self.llm_weight) * c["score"], 4)
            d["reasoning"] = rank.reasoning
            out.append(d)
        out.sort(key=lambda x: -x["combined_score"])
        return out[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第 7 步重排冒烟：融合 top20 -> LLM 逐块打分 -> top-N（含前后位次对比）")
    ap.add_argument("question")
    ap.add_argument("--batch", type=int, default=10, help="每 prompt 块数（2=原版粒度）")
    ap.add_argument("--llm-w", type=float, default=0.7, help="LLM 分权重")
    ap.add_argument("--k", type=int, default=30, help="每路召回条数（透传检索）")
    ap.add_argument("--cand", type=int, default=20, help="融合候选数（重排输入）")
    ap.add_argument("--top-n", type=int, default=6, help="重排输出条数")
    ap.add_argument("--cap", type=int, default=1500, help="每块截断 token 数")
    ap.add_argument("--show", type=int, default=6, help="打印重排后 top 几")
    args = ap.parse_args()

    router = Router(DATA_DIR / "pdf_metadata.csv", DATA_DIR / "parsed" / "chunked")
    doc_ids = router.route(args.question)
    if not doc_ids:
        print("路由失败：问题里没匹配到任何已索引文档的公司名/stem", flush=True)
        sys.exit(1)
    note = "（比较题多公司：当前取第一家演示，改写与汇总在第 8 步）" \
        if len(doc_ids) > 1 else ""
    print(f"路由 -> {doc_ids}{note}", flush=True)

    from enterprise_rag.indexing.ingestor import make_embedder
    retriever = Retriever(embedder=make_embedder())
    cands = retriever.retrieve(args.question, doc_ids[0], k=args.k, top_n=args.cand)
    print(f"\n== 融合 top{len(cands)}（重排输入，截选前 8） ==", flush=True)
    for i, c in enumerate(cands[:8], 1):
        print(f" {i:2d}. s={c['score']:.3f} [{c['type']}:p{c['page']}] "
              f"{_snippet(c['text'])}", flush=True)

    reranker = Reranker(llm_weight=args.llm_w, batch=args.batch, cap=args.cap)
    top = reranker.rerank(args.question, cands, top_n=args.top_n)

    fused_rank = {c["chunk_id"]: i for i, c in enumerate(cands, 1)}
    print(f"\n== 重排后 top{min(args.show, len(top))}"
          f"（llm_w={args.llm_w}，#n=融合位次） ==", flush=True)
    for i, c in enumerate(top[:args.show], 1):
        prev = fused_rank.get(c["chunk_id"])
        delta = "=" if prev == i else (f"↑{prev}" if (prev or 0) > i else f"↓{prev}")
        print(f" {i:2d}. combined={c['combined_score']:.3f} llm={c['relevance_score']:.1f} "
              f"s={c['score']:.3f} #{delta} [{c['type']}:p{c['page']}] "
              f"{_snippet(c['text'], 70)}", flush=True)
        print(f"     └─ {_snippet(c['reasoning'], 110)}", flush=True)

    parents = retriever.parent_context(doc_ids[0], top)
    total = sum(len(p["text"]) for p in parents)
    print(f"\n== 父文档（重排 top{len(top)} -> pages，{len(parents)} 段共 {total} 字符）: "
          f"{' '.join('p' + str(p['page']) for p in parents)} ==", flush=True)
    print(f"\n# 重排 tokens: 入{reranker.tokens_in} / 出{reranker.tokens_out}", flush=True)


if __name__ == "__main__":
    main()
