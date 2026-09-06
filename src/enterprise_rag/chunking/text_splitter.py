"""文本切分：结构化文档 → 子块 + 父块。

检索用子块（小而准），送入 LLM 用其所属父块（大而全）——即 parent document
retrieval。理想实现基于 Docling 的标题层级切分；当前先提供朴素版本跑通链路。

TODO(实现):
- split_parent_child: 按标题层级生成父块，父块内再滑窗切成子块
- 子块记录 parent_id，检索命中子块后回溯父块
"""

from __future__ import annotations


def split_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> list[str]:
    """朴素切分：按空行分段，贪心合并到 max_chars，相邻块保留 overlap。

    仅作为占位实现保证链路可跑，后续由父文档切分替代。
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # 单段超长时硬切
        while len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap_chars :]
        if not para:
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > max_chars:
            chunks.append(current)
            # 用尾部 overlap 起头新块，保证跨块语义连续
            current = current[-overlap_chars:] + "\n\n" + para if current else para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks
