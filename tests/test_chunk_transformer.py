from __future__ import annotations

import json
from types import SimpleNamespace

from enterprise_rag.processing.chunk_transformer import (
    extract_numeric_tokens,
    transform_document,
    validate_transformed_text,
)


class FakeCompletions:
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


class FakeClient:
    def __init__(self, outcomes: list[str | Exception]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


def _content(chunk_id: int, text: str, **extra) -> dict:
    return {
        "id": chunk_id,
        "type": "content",
        "page": 1,
        "text": text,
        "length_tokens": 10,
        "els": [chunk_id, chunk_id + 1],
        **extra,
    }


def test_extract_numeric_tokens_covers_financial_formats() -> None:
    assert extract_numeric_tokens("FY2023 revenue was $1,234.50, up 8%") == {
        "2023",
        "1,234.50",
        "8%",
    }


def test_validation_rejects_missing_numeric_tokens() -> None:
    assert validate_transformed_text(
        "Revenue was $1,234.50", "Revenue improved"
    ) == (False, "numeric_tokens_missing: 1,234.50")


def test_validation_accepts_preserved_numeric_tokens() -> None:
    assert validate_transformed_text(
        "Revenue was $1,234.50", "Revenue was $1,234.50"
    ) == (True, None)


def test_successful_transform_is_auditable_and_skips_non_prose() -> None:
    client = FakeClient([json.dumps({"transformed_text": "Revenue was $1,234."})])
    table = {
        "id": 1,
        "type": "serialized_table",
        "page": 1,
        "text": "Table 2023: $99",
        "length_tokens": 5,
        "els": [1, 2],
    }
    doc = {"metainfo": {}, "pages": [], "chunks": [_content(0, "Header\nRevenue was $1,234."), table]}

    result = transform_document(doc, client=client, model="fake", max_workers=1)

    transformed = result["chunks"][0]
    assert transformed["text"] == "Revenue was $1,234."
    assert transformed["original_text"] == "Header\nRevenue was $1,234."
    assert transformed["length_tokens"] > 0
    assert transformed["transform"]["status"] == "transformed"
    assert transformed["transform"]["model"] == "fake"
    assert result["chunks"][1] == table
    assert len(client.chat.completions.calls) == 1


def test_numeric_loss_falls_back_to_original_text() -> None:
    client = FakeClient([json.dumps({"transformed_text": "Revenue increased."})])
    original = "Revenue increased to $1,234 in 2023."

    result = transform_document(
        {"chunks": [_content(0, original)]}, client=client, model="fake", max_workers=1
    )

    chunk = result["chunks"][0]
    assert chunk["text"] == original
    assert "original_text" not in chunk
    assert chunk["transform"]["status"] == "validation_failed"
    assert "1,234" in chunk["transform"]["reason"]


def test_api_failure_is_isolated_after_retries() -> None:
    client = FakeClient([RuntimeError("offline")] * 3)
    original = "Ordinary prose."

    result = transform_document(
        {"chunks": [_content(0, original)]}, client=client, model="fake", max_workers=1
    )

    chunk = result["chunks"][0]
    assert chunk["text"] == original
    assert chunk["transform"]["status"] == "failed"
    assert "offline" in chunk["transform"]["reason"]
    assert len(client.chat.completions.calls) == 3


def test_successfully_transformed_chunk_is_skipped_on_restart() -> None:
    client = FakeClient([])
    chunk = _content(
        0,
        "Already clean.",
        original_text="Before cleanup.",
        transform={"status": "transformed", "model": "fake"},
    )

    result = transform_document(
        {"chunks": [chunk]}, client=client, model="fake", max_workers=1
    )

    assert result["chunks"][0] == chunk
    assert client.chat.completions.calls == []


def test_prompt_contains_bounded_adjacent_prose_context() -> None:
    client = FakeClient(
        [
            json.dumps({"transformed_text": "Previous context."}),
            json.dumps({"transformed_text": "Current text."}),
            json.dumps({"transformed_text": "Next context."}),
        ]
    )
    doc = {
        "chunks": [
            _content(0, "Previous context."),
            _content(1, "Current text."),
            _content(2, "Next context."),
        ]
    }

    transform_document(doc, client=client, model="fake", max_workers=1)

    current_prompt = client.chat.completions.calls[1]["messages"][1]["content"]
    assert "Previous context." in current_prompt
    assert "Current text." in current_prompt
    assert "Next context." in current_prompt
