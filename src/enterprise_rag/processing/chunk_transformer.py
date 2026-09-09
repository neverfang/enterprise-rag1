"""可选的正文 chunk LLM 清洗与上下文增强。"""

from __future__ import annotations

import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")
_NUMBER = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_CONTEXT_LIMIT = 800

SYSTEM_PROMPT = """You clean prose chunks extracted from an annual-report PDF.
Return valid JSON with exactly one key: {"transformed_text": "..."}.

Rules:
- Clean only the CURRENT CHUNK. Adjacent chunks are context, not content to copy.
- Remove residual headers, footers, control artifacts, and obvious garbling.
- Repair a boundary-cut sentence only when adjacent context directly supports it.
- Retain useful heading breadcrumbs and every fact, name, date, amount, percentage,
  and other number from the current chunk.
- Never invent facts, summarize away details, or add unrelated adjacent content.
"""


def _numeric_tokens_in_order(text: str) -> list[str]:
    return [match.group(0) for match in _NUMBER.finditer(text)]


def extract_numeric_tokens(text: str) -> set[str]:
    """提取需要在清洗结果中守恒的数字 token。"""
    return set(_numeric_tokens_in_order(text))


def validate_transformed_text(original: str, transformed: str) -> tuple[bool, str | None]:
    """拒绝空输出或丢失原始数字的 LLM 结果。"""
    if not transformed.strip():
        return False, "empty_transformed_text"
    transformed_tokens = extract_numeric_tokens(transformed)
    missing = [
        token for token in _numeric_tokens_in_order(original)
        if token not in transformed_tokens
    ]
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        return False, f"numeric_tokens_missing: {', '.join(unique_missing)}"
    return True, None


def _adjacent_content(chunks: list[dict], index: int, direction: int) -> str:
    cursor = index + direction
    while 0 <= cursor < len(chunks):
        candidate = chunks[cursor]
        if candidate.get("type") == "content":
            text = candidate.get("text") or ""
            return text[-_CONTEXT_LIMIT:] if direction < 0 else text[:_CONTEXT_LIMIT]
        cursor += direction
    return ""


def _user_prompt(chunks: list[dict], index: int) -> str:
    previous = _adjacent_content(chunks, index, -1) or "(none)"
    current = chunks[index].get("text") or ""
    following = _adjacent_content(chunks, index, 1) or "(none)"
    return (
        f"PREVIOUS PROSE CONTEXT:\n{previous}\n\n"
        f"CURRENT CHUNK:\n{current}\n\n"
        f"NEXT PROSE CONTEXT:\n{following}"
    )


def _transform_one(
    chunks: list[dict],
    index: int,
    client,
    model: str,
    retries: int,
) -> tuple[int, dict]:
    original_chunk = chunks[index]
    original_text = original_chunk.get("text") or ""
    last_error: Exception | None = None
    for _attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _user_prompt(chunks, index)},
                ],
                timeout=120,
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            transformed = payload.get("transformed_text")
            if not isinstance(transformed, str):
                raise TypeError("响应缺少 transformed_text 字符串")
            valid, reason = validate_transformed_text(original_text, transformed)
            if not valid:
                chunk = copy.deepcopy(original_chunk)
                chunk["transform"] = {
                    "status": "validation_failed",
                    "model": model,
                    "reason": reason,
                }
                return index, chunk
            chunk = copy.deepcopy(original_chunk)
            chunk["original_text"] = original_text
            chunk["text"] = transformed.strip()
            chunk["length_tokens"] = len(_ENC.encode(chunk["text"]))
            chunk["transform"] = {"status": "transformed", "model": model}
            return index, chunk
        except (AttributeError, IndexError, json.JSONDecodeError, TypeError, ValueError,
                RuntimeError) as exc:
            last_error = exc

    chunk = copy.deepcopy(original_chunk)
    chunk["transform"] = {
        "status": "failed",
        "model": model,
        "reason": str(last_error or "unknown transform error"),
    }
    return index, chunk


def transform_document(
    doc: dict,
    *,
    client,
    model: str,
    max_workers: int = 4,
    retries: int = 3,
) -> dict:
    """清洗一份 chunked 文档；单块失败不会中断其他块。"""
    if max_workers <= 0:
        raise ValueError("max_workers 必须大于 0")
    if retries <= 0:
        raise ValueError("retries 必须大于 0")

    result = copy.deepcopy(doc)
    chunks = result.get("chunks", [])
    pending = [
        index for index, chunk in enumerate(chunks)
        if chunk.get("type") == "content"
        and (chunk.get("transform") or {}).get("status") != "transformed"
    ]
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _transform_one, chunks, index, client, model, retries
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                index, transformed = future.result()
                chunks[index] = transformed

    counts: dict[str, int] = {}
    for chunk in chunks:
        status = (chunk.get("transform") or {}).get("status")
        if status:
            counts[status] = counts.get(status, 0) + 1
    result.setdefault("metainfo", {})["chunk_transform"] = {
        "model": model,
        "statuses": counts,
    }
    return result
