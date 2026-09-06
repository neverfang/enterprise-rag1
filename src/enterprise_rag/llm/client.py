"""LLM / Embedding API 封装（OpenAI 兼容接口）。

所有对 LLM 的调用都必须经过这里，便于统一限速、重试、计费统计。
参考原项目 api_request_parallel_processor.py 的并发批处理思路。

TODO(实现):
- get_client(): OpenAI(base_url 取自 .env，api_key 取自 .env)
- chat(): 单次对话，支持重试
- embed(): 批量嵌入
- 并发批处理 + 限速（大文档集嵌入时需要）
"""

from __future__ import annotations

from openai import OpenAI

from enterprise_rag.config import get_api_key, get_base_url

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """懒加载的全局客户端。"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_api_key(), base_url=get_base_url())
    return _client
