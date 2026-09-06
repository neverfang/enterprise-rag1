"""端到端冒烟测试：用 data/test_set/ 的小样例跑通全流水线。

改完任何模块后运行：python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise_rag.config import DATA_DIR, load_config  # noqa: E402


def main() -> None:
    config = load_config("default")
    test_set = DATA_DIR / "test_set"
    pdfs = list(test_set.glob("*.pdf"))
    questions_file = test_set / "questions.jsonl"

    print(f"配置加载 OK: {list(config.keys())}")
    print(f"测试 PDF 数量: {len(pdfs)}（目录: {test_set}）")
    print(f"问题文件存在: {questions_file.exists()}")

    # TODO: 依次跑 parse → ingest → ask（首个问题），全链路通过后退出码 0
    raise SystemExit("冒烟测试待实现：放入测试 PDF 后补全本脚本")


if __name__ == "__main__":
    main()
