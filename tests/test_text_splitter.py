from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from enterprise_rag.processing.text_splitter import split_all


class _Completions:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"transformed_text": "Clean paragraph."})
                    )
                )
            ]
        )


def _write_serialized_doc(input_dir: Path) -> None:
    input_dir.mkdir()
    doc = {
        "metainfo": {"doc_id": "sample"},
        "content": [
            {"page": 1, "type": "paragraph", "text": "Noisy paragraph.", "level": 0}
        ],
        "tables": [],
    }
    (input_dir / "sample.json").write_text(
        json.dumps(doc), encoding="utf-8"
    )


def test_split_all_does_not_transform_by_default(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_serialized_doc(input_dir)

    split_all(input_dir, output_dir, 300, 50)

    result = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert result["chunks"][0]["text"] == "Noisy paragraph."
    assert "transform" not in result["chunks"][0]
    assert "chunk_transform" not in result["metainfo"]


def test_split_all_transforms_only_when_explicitly_enabled(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_serialized_doc(input_dir)
    completions = _Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    split_all(
        input_dir,
        output_dir,
        300,
        50,
        transform=True,
        client=client,
        model="fake",
        transform_workers=1,
    )

    result = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert result["chunks"][0]["text"] == "Clean paragraph."
    assert result["chunks"][0]["transform"]["status"] == "transformed"
    assert result["metainfo"]["chunk_transform"]["model"] == "fake"
    assert completions.calls == 1
