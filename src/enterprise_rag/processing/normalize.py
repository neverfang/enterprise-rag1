"""解析产物文本整理：控制字符清理、连字断痕修复、空白规范化。

处理第 2 步质量体检发现的遗留问题：
- 控制字符：文本层残留的 U+0003/U+0004 等脚注符号（如 CrossFirst 共 29 个）
- 连字断痕：文本层把 fi/fl 连字提取成分离字符，"fiscal" -> "fi scal"、
  "five" -> "fi ve"（Air Products 实测存在）
- 非断行空格、行尾空白、连续空白

纯函数模块，被 table_serializer 的批处理调用。
"""

from __future__ import annotations

import re

# C0 控制字符（保留换行 chr(10)）+ DEL + 零宽空格 + BOM。
# 用 chr() 逐个构造，避免源码里出现转义序列或不可见字符。
_CONTROL_CHARS = "".join(
    chr(c) for c in list(range(10)) + list(range(11, 32)) + [127, 0x200B, 0xFEFF]
)
_CTRL = re.compile("[" + re.escape(_CONTROL_CHARS) + "]")
# 连字断痕：词首 fi/fl + 空格 + 小写字母（"fi scal" -> "fiscal"）
_LIGATURE = re.compile(r"\b(fi|fl) (?=[a-z])")
# 行尾空白 / 连续空格 / 3 行以上空行
_TRAIL_WS = re.compile(r"[ \t]+\n")
_MULTI_SP = re.compile(r"[ \t]{2,}")
_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """清洗单个文本块（幂等，可重复应用）。"""
    if not text:
        return text
    text = _CTRL.sub("", text)
    text = _LIGATURE.sub(r"\1", text)
    text = text.replace(chr(0xA0), " ")  # 非断行空格
    text = _TRAIL_WS.sub("\n", text)
    text = _MULTI_SP.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return text.strip()


def normalize_doc(doc: dict) -> dict:
    """对解析产物 JSON 原地应用规范化：content 各元素文本 + 表格 HTML。"""
    for c in doc.get("content", []):
        if c.get("text"):
            c["text"] = normalize_text(c["text"])
    for t in doc.get("tables", []):
        if t.get("html"):
            # MinerU 的 table_body 是单行 HTML，strip/替换空白不影响标签结构
            t["html"] = normalize_text(t["html"])
        if t.get("caption"):
            t["caption"] = normalize_text(t["caption"])
    return doc
