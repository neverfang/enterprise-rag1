"""检索：query → 嵌入 → 余弦 top-k 子块 → 回溯父块。

TODO(实现):
- 向量检索 top_k（配置 retrieval.top_k）
- 命中子块按 parent_id 去重回溯父块（parent_window 控制相邻扩展）
- 返回统一结构：[{parent_text, child_text, doc_id, score, page}]
"""

from __future__ import annotations

from typing import Any


def retrieve(query: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """检索并回溯父块。"""
    raise NotImplementedError("retriever.retrieve 待实现")
