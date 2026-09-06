"""索引构建：子块 → 嵌入向量 → 本地向量索引。

第一版用 numpy：嵌入矩阵存 npz、块元数据存 jsonl，放在 data/parsed/index/。
接口设计成可替换（ingest / search 两个函数），后续可无痛换 Qdrant。

TODO(实现):
- 读取 data/parsed/docs/ 全部文档 → 切分 → 批量嵌入（batch_size 取配置）
- 保存 embeddings.npz + chunks.jsonl（含 chunk_id、parent_id、text、doc_id、页码）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def ingest(chunks_path: Path, config: dict[str, Any]) -> Path:
    """为切分后的块建索引，返回索引目录。"""
    raise NotImplementedError("ingest.ingest 待实现")
