"""答案生成：重排序后的上下文 + 问题 → 结构化答案。

TODO(实现):
- 用 prompts.ANSWER_PROMPT 组装上下文（标注来源公司/页码）
- 解析 <thought>/<answer> 标签，answer 部分返回给调用方
"""

from __future__ import annotations

from typing import Any


def generate_answer(query: str, contexts: list[dict[str, Any]], config: dict[str, Any]) -> str:
    """生成最终答案。"""
    raise NotImplementedError("answer.generate_answer 待实现")
