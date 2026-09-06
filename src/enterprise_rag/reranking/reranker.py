"""LLM 重排序：对初检父块按与 query 的相关度重新打分排序。

TODO(实现):
- 用 prompts.RERANK_PROMPT 让 LLM 输出块序号的相关度排序（JSON）
- 取前 top_n（配置 reranking.top_n）
- reranking.enabled=false 时直接透传
"""

from __future__ import annotations

from typing import Any


def rerank(query: str, contexts: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """重排序，返回 top_n 上下文。"""
    raise NotImplementedError("reranker.rerank 待实现")
