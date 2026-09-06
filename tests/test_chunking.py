"""chunking 模块单测：从最容易测的纯逻辑开始。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.chunking.text_splitter import split_text  # noqa: E402


def test_split_text_short():
    """短文本不切分，原样返回单块。"""
    assert split_text("只有一段话", max_chars=100) == ["只有一段话"]


def test_split_text_respects_max_chars():
    """切分后每块不超过 max_chars（单段超长的硬切除外，同样受限）。"""
    paras = ["A" * 50] * 10  # 每段 50 字符，共 500
    text = "\n\n".join(paras)
    chunks = split_text(text, max_chars=120, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 120 for c in chunks)


def test_split_text_no_content_loss():
    """切分不丢失实质内容（overlap 造成的重复不计）。"""
    text = "\n\n".join(f"第{i}段内容" for i in range(20))
    chunks = split_text(text, max_chars=30, overlap_chars=5)
    merged = "".join(chunks)
    for i in range(20):
        assert f"第{i}段内容" in merged
