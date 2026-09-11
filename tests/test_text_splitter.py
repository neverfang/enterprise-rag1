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


def _write_doc_with_table(input_dir: Path) -> None:
    input_dir.mkdir()
    doc = {
        "metainfo": {"doc_id": "sample"},
        "content": [
            {"page": 1, "type": "paragraph", "text": "Intro paragraph.", "level": 0},
            {"page": 1, "type": "table", "table_id": "t0", "text": "fallback text"},
            # 空占位表：无 caption/html/text，序列化信息块为空（MinerU 实测存在）
            {"page": 1, "type": "table", "table_id": "t1", "text": ""},
        ],
        "tables": [
            {
                "id": "t0",
                "page": 1,
                "html": "<table><tr><td>100</td></tr></table>",
                "caption": "Revenue",
                "serialized": {
                    "subject_core_entities_list": ["Revenue"],
                    "relevant_headers_list": ["2022"],
                    "information_blocks": [
                        {"information_block": "Revenue for 2022 is 100 thousand."}
                    ],
                },
            },
            {
                "id": "t1",
                "page": 1,
                "html": "",
                "caption": "",
                "serialized": {
                    "subject_core_entities_list": [],
                    "relevant_headers_list": [],
                    "information_blocks": [],
                },
            },
        ],
    }
    (input_dir / "sample.json").write_text(json.dumps(doc), encoding="utf-8")


def _table_chunk(result: dict) -> dict:
    return next(c for c in result["chunks"] if c["type"] == "serialized_table")


def test_split_all_uses_serialized_blocks_by_default(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_doc_with_table(input_dir)

    split_all(input_dir, output_dir, 300, 50)

    result = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    assert "Revenue for 2022 is 100 thousand." in _table_chunk(result)["text"]
    assert "<table>" not in _table_chunk(result)["text"]
    # 空占位表不建块（防空块稀释索引、卡 ingestor 的非空校验）
    assert len([c for c in result["chunks"] if c["type"] == "serialized_table"]) == 1
    assert all(c["text"].strip() for c in result["chunks"])


def test_split_all_table_source_html_keeps_raw_html(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_doc_with_table(input_dir)

    split_all(input_dir, output_dir, 300, 50, table_source="html")

    result = json.loads((output_dir / "sample.json").read_text(encoding="utf-8"))
    body = _table_chunk(result)["text"]
    assert "Table: Revenue" in body
    assert "<table><tr><td>100</td></tr></table>" in body
    assert "Revenue for 2022" not in body  # 序列化信息块不混入
    assert "<table>" in result["pages"][0]["text"]  # 页父文本同步用原始 HTML


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
