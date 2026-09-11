"""RAGAS 评测：faithfulness / answer relevancy / context precision。

对 pipeline 的 answers_run*.json（2026-09-11 起含 contexts 字段）跑三个
社区标准指标，量化"答案忠于上下文 / 答案切题 / 检索上下文质量"。

口径（务必随数字一起报告）：
- 只评**双方都非 N/A** 的题（系统作答且应有答案）——答案为 N/A 的题
  faithfulness 无 claim 可验、answer relevancy 必然失真（N/A 不含问题词），
  "敢于回 N/A 不编造"由端到端准确率的 N/A 子项覆盖，不在 RAGAS 里重复
- ground_truth 用 answers_eval 首个可接受答案（context precision 用）
- contexts 截断：每题总字符 cap（默认 16K，保序弃尾——重排后的父文本
  顺序即重要性序），控制 judge 成本
- judge = DeepSeek（.env 同生成端 LLM_*）；embedding = 硅基流动 bge-m3
  （answer relevancy 反向生成问题用），与建库同模型
- 依赖：pyproject 的 [metrics] 可选组（ragas 0.4 + langchain 0.3 生态）

用法：
    python -m enterprise_rag.eval_ragas <answers_run.json> \
        [--gt data/answers_eval.json] [--cap-chars 16000] [--limit N] \
        [--out data/eval_ragas_result.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from enterprise_rag.evaluate import _is_na, parse_answers

# src/enterprise_rag/eval_ragas.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _make_judges():
    """judge LLM（DeepSeek）与 embedding（硅基流动 bge-m3），均来自 .env。

    用 ragas 的经典路径（LangchainLLMWrapper + ragas.metrics 实例）：0.4 的
    evaluate() 仍只认老 Metric 体系（collections 新指标不经过 evaluate），
    老路径 deprecated 至 v1.0 才移除——pyproject 已锁 ragas<1.0。"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise SystemExit("请先在 .env 配置 DEEPSEEK_API_KEY")
    emb_key = os.environ.get("EMBED_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
    if not emb_key:
        raise SystemExit("请先在 .env 配置 EMBED_API_KEY")
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=0, timeout=120, max_retries=0))
    emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
        api_key=emb_key,
        base_url=os.environ.get("EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
        chunk_size=32))               # 硅基流动单请求条数上限 32（同 ingestor）
    return llm, emb


def _truncate(contexts: list[str], cap: int) -> list[str]:
    out, total = [], 0
    for c in contexts:                 # 保序弃尾：父文本顺序即重要性序
        if total + len(c) > cap:
            break
        out.append(c)
        total += len(c)
    return out or contexts[:1]         # 单段超 cap 时至少保一段


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGAS 三指标评测")
    ap.add_argument("answers", type=Path, help="pipeline 输出（须含 contexts）")
    ap.add_argument("--gt", type=Path, default=DATA_DIR / "answers_eval.json")
    ap.add_argument("--cap-chars", type=int, default=16_000, help="每题 contexts 截断")
    ap.add_argument("--limit", type=int, help="最多评 N 题（冒烟用）")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eval_ragas_result.json")
    args = ap.parse_args()

    runs = json.loads(args.answers.read_text(encoding="utf-8"))
    gt = json.loads(args.gt.read_text(encoding="utf-8"))

    rows = []                          # 双方都非 N/A 且有 contexts 的题
    n_no_ctx = n_na = 0
    for a in runs:
        meta = gt.get(a["question"])
        if not meta:
            continue
        gts = parse_answers(meta["answers"])
        pred_na, gt_na = _is_na(a.get("final_answer")), all(_is_na(g) for g in gts)
        if pred_na or gt_na:
            n_na += 1
            continue
        if not a.get("contexts"):
            n_no_ctx += 1
            continue
        rows.append({
            "user_input": a["question"],
            "retrieved_contexts": _truncate(a["contexts"], args.cap_chars),
            "response": str(a.get("final_answer")),
            "reference": str(gts[0]),
        })
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"可评题：{len(rows)}（跳过 N/A 口径 {n_na}、无 contexts {n_no_ctx}）")
    if not rows:
        return

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (  # noqa: F401 —— 经典路径实例（见 _make_judges 注）
        answer_relevancy, context_precision, faithfulness)

    llm, emb = _make_judges()
    answer_relevancy.strictness = 1  # DeepSeek 不支持 n>1 采样，1 个反向问题即评
    result = evaluate(
        dataset=EvaluationDataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm, embeddings=emb, show_progress=True)

    df = result.to_pandas()
    cols = {m: c for m, c in [("faithfulness", "faithfulness"),
                              ("answer_relevancy", "answer_relevancy"),
                              ("context_precision", "context_precision")]
            if c in df.columns}
    print(f"\n== RAGAS（{len(df)} 题均值；judge=DeepSeek，embedding=bge-m3） ==")
    means = {}
    for metric, col in cols.items():
        means[metric] = float(df[col].mean())
        print(f"  {metric:<20} {means[metric]:.3f}")

    df.to_json(args.out, orient="records", force_ascii=False, indent=1)
    print(f"每题明细已存 {args.out}")


if __name__ == "__main__":
    main()
