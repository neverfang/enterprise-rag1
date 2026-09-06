"""表格 LLM 序列化：HTML 表格 -> 上下文独立的信息块。

参考 IlyaRice/RAG-Challenge-2 的 src/tables_serialization.py（原版用 gpt-4o-mini，
本项目用 DeepSeek，OpenAI 兼容接口）。核心思想：表格行裸文本（如
"Sales | 12,699 | 10,323"）切成 chunk 后既无公司名也无单位币种，无法被问题
独立命中；因此对每个表取同页前后正文做上下文，连同表格 HTML 一起发给 LLM
（temperature=0，JSON 输出），改写成若干自描述信息块——带全表头、单位、
币种、脚注、表名、公司与报告期，后续（第 4-5 步）直接作为 chunk 进向量库。

数据流：data/parsed/docs/*.json ->（normalize 整理 + 表格序列化）
       data/parsed/serialized/*.json（第 2 步产物保持不动）
断点续跑：已带 serialized 字段的表跳过，按表粒度续。

用法：
    python -m enterprise_rag.processing.table_serializer [input_dir] \
        [--out DIR] [--workers N] [--limit N（每份文档最多序列化 N 表，冒烟用）]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from enterprise_rag.processing.normalize import normalize_doc

# src/enterprise_rag/processing/table_serializer.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

# —— 提示词：忠实沿用原项目的系统提示与 schema 说明，仅追加 JSON 输出格式要求 ——

SYSTEM_PROMPT = (
    "You are a table serialization agent.\n"
    "Your task is to create a set of contextually independent blocks of information "
    "based on the provided table and surrounding text.\n"
    "These blocks must be totally context-independent because they will be used as "
    "separate chunks to populate a database."
)

JSON_FORMAT_INSTRUCTION = """
Return only valid json (no markdown fences) of exactly this shape:
{
  "subject_core_entities_list": ["...", ...],
  "relevant_headers_list": ["...", ...],
  "information_blocks": [
    {"subject_core_entity": "...", "information_block": "..."}
  ]
}
- subject_core_entities_list: complete list of core entities. Usually each row header
  represents a core entity. Empty headers are possible too - interpret and list them
  as well (usually it is a total or something similar).
- relevant_headers_list: a list of ALL headers relevant to the subjects. These headers
  will serve as keys in each information block. In most cases each column header
  represents a core entity.
- information_blocks: fully described context-independent information blocks.
  Each information_block MUST include:
  1. All related header information
  2. All related units and their descriptions
     (if a header is "Total", always write additional context about what this total
     represents in this block!)
  3. All additional info for context enrichment that is present in the table or the
     surrounding text, to make the block completely context-independent:
     - The name of the table and its caption
     - Additional footnotes
     - The currency used
     - The way amounts are presented
     - The company the report belongs to, and the reporting period
  If one row of the table does not make sense without neighboring rows, merge
  information from neighboring rows into one block.
  SKIPPING ANY VALUABLE INFORMATION WILL BE HEAVILY PENALIZED!
"""


class InformationBlock(BaseModel):
    subject_core_entity: str
    information_block: str


class TableBlocks(BaseModel):
    subject_core_entities_list: list[str] = Field(default_factory=list)
    relevant_headers_list: list[str] = Field(default_factory=list)
    information_blocks: list[InformationBlock]


def make_client() -> tuple[OpenAI, str]:
    """从 .env 读取配置，返回 (OpenAI 客户端, 模型名)。默认 DeepSeek。"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key or "填" in api_key:
        raise SystemExit("请先在项目根目录 .env 里配置 DEEPSEEK_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    return OpenAI(api_key=api_key, base_url=base_url), model


def _join_blocks(blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("text"))


def _get_table_context(doc: dict, table_id: str) -> tuple[str, str]:
    """取表格所在页、相邻表格之间的正文块作为前后文（对应原版 _get_table_context）。

    我们的内容流是带页码的扁平列表（原版是按页分组的 content），先按页切片再定位。
    """
    table = next(t for t in doc["tables"] if t["id"] == table_id)
    page = table.get("page")
    if page is None:
        return "", ""
    blocks = [c for c in doc["content"] if c.get("page") == page]
    pos = next(
        (i for i, b in enumerate(blocks)
         if b.get("type") == "table" and b.get("table_id") == table_id),
        -1,
    )
    if pos == -1:
        return "", ""
    prev = max(
        (i for i, b in enumerate(blocks) if b.get("type") == "table" and i < pos),
        default=-1,
    )
    nxt = next(
        (i for i, b in enumerate(blocks) if b.get("type") == "table" and i > pos),
        -1,
    )
    before = _join_blocks(blocks[prev + 1:pos])
    # 后文止于下一张表前一格（那一格通常是下一张表的 caption，归属下一张表）
    after_end = (nxt - 1) if nxt != -1 else min(pos + 4, len(blocks))
    after = _join_blocks(blocks[pos + 1:after_end])
    return before, after


def _build_query(doc: dict, table: dict, before: str, after: str) -> str:
    parts = []
    # 公司名直接注入（原版只靠页面上下文，上下文稀薄的表会丢失公司归属）
    company = (doc.get("metainfo") or {}).get("company")
    if company:
        parts.append(
            f"This table comes from the annual report of {company}.")
    if before:
        parts.append(
            'Here is additional text before the table that might be relevant (or not):\n'
            '"""' + before + '"""'
        )
    caption = (table.get("caption") or "").strip()
    parts.append(
        ("Table caption: " + caption + "\n" if caption else "")
        + 'Here is a table in HTML format:\n"""' + (table.get("html") or "") + '"""'
    )
    if after:
        parts.append(
            'Here is additional text after the table that might be relevant (or not):\n'
            '"""' + after + '"""'
        )
    # 超大表（如 70+ 行的明细表）逐行成块会超出 max_tokens 导致 JSON 截断，
    # 要求合并相关行压缩块数（实测 dev 集仅 2 张此类表）
    html = table.get("html") or ""
    if html.count("<tr>") > 50 or len(html) > 12000:
        parts.append(
            "Note: this table is very large. To keep the output within limits, "
            "merge related rows into grouped information blocks (aim for at most "
            "40 blocks in total) and keep each block compact.")
    return "\n\n".join(parts)


def serialize_table(client: OpenAI, model: str, table_id: str,
                    query: str) -> tuple[dict, tuple[int, int]]:
    """序列化单个表（3 次重试），返回 (TableBlocks dict, (输入tokens, 输出tokens))。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + JSON_FORMAT_INSTRUCTION},
        {"role": "user", "content": query},
    ]
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=8192,
                response_format={"type": "json_object"},  # DeepSeek JSON 输出模式
                messages=messages,
                timeout=180,
            )
            data = json.loads(resp.choices[0].message.content)
            blocks = TableBlocks.model_validate(data)
            usage = (resp.usage.prompt_tokens, resp.usage.completion_tokens) \
                if resp.usage else (0, 0)
            return blocks.model_dump(), usage
        except Exception as e:  # noqa: BLE001 —— 重试循环就是要兜住一切可重试失败
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"表 {table_id} 已重试 3 次: {last_err}")


def process_doc(client: OpenAI, model: str, in_path: Path, out_path: Path,
                workers: int = 8, limit: int | None = None) -> None:
    """整理 + 序列化一份文档并落盘（输出已存在则以它为底本续跑）。"""
    base_path = out_path if out_path.exists() else in_path
    doc = json.loads(Path(base_path).read_text(encoding="utf-8"))
    normalize_doc(doc)  # 幂等，续跑时重复应用无副作用

    tables = doc.get("tables", [])
    pending = [t for t in tables if not t.get("serialized")]
    already = len(tables) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"[SKIP] {in_path.name}（{already} 表均已序列化）")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        out_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    _save()  # 先落规范化后的底本，序列化结果随完成逐次写回

    queries = {
        t["id"]: _build_query(doc, t, *_get_table_context(doc, t["id"]))
        for t in pending
    }
    failed: list[str] = []
    tokens = [0, 0]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(serialize_table, client, model, tid, q): tid
            for tid, q in queries.items()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            tid = futs[fut]
            table = next(t for t in doc["tables"] if t["id"] == tid)
            try:
                result, (pt, ct) = fut.result()
                table["serialized"] = result
                tokens[0] += pt
                tokens[1] += ct
            except Exception as e:  # noqa: BLE001 —— 单表失败不中断整批，留待续跑
                failed.append(tid)
                print(f"  [FAIL] {in_path.stem} {tid}: {e}")
            if i % 10 == 0 or i == len(pending):
                _save()
    print(
        f"[OK] {in_path.name}: 新序列化 {len(pending) - len(failed)}/{len(pending)}，"
        f"已有 {already}，失败 {len(failed)}；"
        f"tokens 入{tokens[0]}/出{tokens[1]}；{time.perf_counter() - t0:.0f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第 3 步：文本整理 + 表格 LLM 序列化（parsed/docs -> parsed/serialized）")
    ap.add_argument("input", type=Path, nargs="?", default=DATA_DIR / "parsed" / "docs",
                    help="输入目录（默认 data/parsed/docs）")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "parsed" / "serialized")
    ap.add_argument("--workers", type=int, default=8, help="并发请求数")
    ap.add_argument("--limit", type=int,
                    help="每份文档最多序列化的表数（冒烟测试用）")
    args = ap.parse_args()

    client, model = make_client()
    print(f"# 模型 {model}，输入 {args.input}，输出 {args.out}，并发 {args.workers}")
    for in_path in sorted(args.input.glob("*.json")):
        process_doc(client, model, in_path, args.out / in_path.name,
                    workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
