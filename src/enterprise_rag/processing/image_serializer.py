"""图片多模态序列化：年报图片 -> 上下文独立的文字转写（图表数据入检索库）。

背景（3.5 步审计结论，2026-09-07）：dev 集 197 张图中仅 41 张（21%）携带数据
（图表/表图/信息图），其余为照片和图标；盲目全量转写既浪费又会污染语料。
故两阶段：先轻量分类（严格两行格式 + thinking 关闭），仅 HAS_DATA 的做完整转写。
转写只收录图上印出的数字（禁止按网格线估值——冒烟实测模型会自己"估"出
似是而非的值），配合 caption/前后文/公司名注入，与表格序列化同一思路。

图片定位：统一 JSON 不存 img_path（第 2 步转换时丢弃），从 mineru_raw 的
{name}_content_list.json 按**顺序位置**对齐 image 条目与 content 流的 picture 元素
（审计验证 10/10 文档数量与页码全对、197/197 文件在盘）。

数据流：data/parsed/serialized/*.json（原地更新，picture 元素挂 vl 字段：
{type, data, text, img}；NO_DATA 的 text 为空串但保留分类结果）
断点续跑：已带 vl 字段的 picture 跳过，按图粒度。

用法：
    python -m enterprise_rag.processing.image_serializer [input_dir] \
        [--raw DIR] [--workers N] [--limit N（每份文档最多处理 N 图，冒烟用）]

注意：
- deepseek-v4-flash-vision-exp 是混合推理模型：thinking 开着且 max_tokens 小时，
  推理会吃光预算返回空 content（审计实测）。分类阶段显式关 thinking；
  转写阶段保持默认但 max_tokens=4096 留足余量
- 图片只能放 user 消息（放 system 会 400）；本地文件用 base64 data URL 最简单
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from enterprise_rag.processing.normalize import normalize_text

# src/enterprise_rag/processing/image_serializer.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

CLASSIFY_PROMPT = (
    "Classify this image from an annual report. Answer in EXACTLY this format "
    "(two lines, nothing else):\n"
    "TYPE: <one of: chart | table_image | infographic | diagram | photo | logo_icon | other>\n"
    "DATA: <HAS_DATA if the image contains numbers, statistics, financial figures, "
    "or data labels; otherwise NO_DATA>"
)

TRANSCRIBE_PROMPT = """You are an image transcription agent for an annual report RAG pipeline.
Your output will be used as standalone retrieval chunks, so it must be fully context-independent.
This image comes from the annual report of {company} (page {page}).
Caption found in the PDF: "{caption}" (may be empty)
Here is additional text before the image that might be relevant (or not):
\"\"\"{before}\"\"\"
Here is additional text after the image that might be relevant (or not):
\"\"\"{after}\"\"\"

Transcribe the image into a compact structured plain-text form (markdown list or table).
Rules:
1. Transcribe ONLY numbers and labels that are actually printed on the image.
   NEVER estimate or guess values from axis positions or gridlines. If a value is not
   printed, describe the trend qualitatively instead of inventing numbers.
2. Include: the chart/table title, all series and category names, units, currency,
   fiscal periods, and every printed data point.
3. Add context enrichment available from the caption or surrounding text: what the
   image is about, the company it belongs to, the reporting period.
4. No analysis, forecasts or commentary beyond transcription.
5. If some text is too small or blurry to read reliably, mark that item as
   [unreadable] rather than guessing."""


def make_client() -> tuple[OpenAI, str]:
    """从 .env 读取配置，返回 (OpenAI 客户端, 视觉模型名)。"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key or "填" in api_key:
        raise SystemExit("请先在项目根目录 .env 里配置 DEEPSEEK_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("LLM_VISION_MODEL", "deepseek-v4-flash-vision-exp")
    return OpenAI(api_key=api_key, base_url=base_url), model


def _vision_call(client: OpenAI, model: str, img_path: Path, prompt: str,
                 *, max_tokens: int, disable_thinking: bool,
                 retries: int = 3) -> tuple[str, tuple[int, int]]:
    """带图调用（3 次重试），返回 (content, (入tokens, 出tokens))。

    content 为空串视为失败（thinking 吃光预算的典型症状），交给重试。
    """
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict = {}
            if disable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                timeout=180,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                **kwargs,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("空 content（thinking 耗尽 max_tokens？）")
            usage = (resp.usage.prompt_tokens, resp.usage.completion_tokens) \
                if resp.usage else (0, 0)
            return content, usage
        except Exception as e:  # noqa: BLE001 —— 重试循环兜住一切可重试失败
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{img_path.name[:16]}… 已重试 {retries} 次: {last_err}")


def classify_image(client: OpenAI, model: str, img_path: Path) -> tuple[str, str, tuple[int, int]]:
    """阶段一：轻量分类，返回 (type, HAS_DATA/NO_DATA, usage)。"""
    text, usage = _vision_call(
        client, model, img_path, CLASSIFY_PROMPT,
        max_tokens=512, disable_thinking=True)
    m = re.search(r"TYPE:\s*\**\s*(\S+)", text, re.IGNORECASE)
    d = re.search(r"DATA:\s*\**\s*(\S+)", text, re.IGNORECASE)
    itype = (m.group(1).strip().lower() if m else "other")
    idata = (d.group(1).strip().upper() if d else "NO_DATA")
    if "HAS" in idata:
        idata = "HAS_DATA"
    else:
        idata = "NO_DATA"
    return itype, idata, usage


def transcribe_image(client: OpenAI, model: str, img_path: Path,
                     query: str) -> tuple[str, tuple[int, int]]:
    """阶段二：完整转写（仅 HAS_DATA），返回 (转写文本, usage)。"""
    return _vision_call(
        client, model, img_path, query,
        max_tokens=4096, disable_thinking=False)


def _align_image_paths(raw_dir: Path, stem: str,
                       n_pics: int) -> list[tuple[Path, str]]:
    """content 流 picture 元素 -> (裁剪图绝对路径, 相对 img_path)，按顺序位置对齐。"""
    cl_files = list((raw_dir / stem).rglob("*_content_list.json"))
    if not cl_files:
        raise RuntimeError(f"{raw_dir / stem} 下未找到 *_content_list.json")
    cl = max(cl_files, key=lambda p: p.stat().st_mtime)
    entries = json.loads(cl.read_text(encoding="utf-8"))
    imgs = [e for e in entries if e.get("type") == "image"]
    if len(imgs) != n_pics:
        raise RuntimeError(
            f"图片对齐失败：content_list {len(imgs)} 张 != 内容流 {n_pics} 张")
    out: list[tuple[Path, str]] = []
    for e in imgs:
        rel = e.get("img_path") or ""
        f = cl.parent / rel
        if not f.exists():
            raise RuntimeError(f"图片文件缺失: {f}")
        out.append((f, rel))
    return out


def _get_image_context(doc: dict, idx: int, window: int = 3) -> tuple[str, str]:
    """取图片所在页、流上前后的正文块作上下文（同页、跳过其他图片元素）。"""
    els = doc["content"]
    page = els[idx].get("page")

    def _txt(e: dict) -> bool:
        return e is not els[idx] and e.get("page") == page and e.get("type") != "picture" \
            and bool((e.get("text") or "").strip())

    before = [normalize_text(e["text"]) for e in reversed(els[:idx]) if _txt(e)][:window]
    after = [normalize_text(e["text"]) for e in els[idx + 1:] if _txt(e)][:window]
    return "\n".join(filter(None, reversed(before))), "\n".join(after)


def _build_query(doc: dict, idx: int) -> str:
    els = doc["content"]
    company = (doc.get("metainfo") or {}).get("company") or "the company"
    before, after = _get_image_context(doc, idx)
    caption = normalize_text(els[idx].get("text") or "")
    return TRANSCRIBE_PROMPT.format(
        company=company, page=els[idx].get("page"), caption=caption or "(none)",
        before=before or "(none)", after=after or "(none)")


def process_doc(client: OpenAI, model: str, in_path: Path, raw_dir: Path,
                workers: int = 8, limit: int | None = None) -> None:
    """对一份文档的全部 picture 元素做两阶段序列化，原地写回。"""
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    pic_idxs = [i for i, c in enumerate(doc["content"]) if c.get("type") == "picture"]
    if not pic_idxs:
        print(f"[SKIP] {in_path.name}（无图片）")
        return
    try:
        aligned = _align_image_paths(raw_dir, in_path.stem, len(pic_idxs))
    except Exception as e:  # noqa: BLE001 —— 对齐失败整份跳过，不误写
        print(f"[FAIL] {in_path.name}: {e}")
        return
    # aligned 与 pic_idxs 一一对应（同为全量按序）；只处理未带 vl 的
    pending = [(idx, aligned[k]) for k, idx in enumerate(pic_idxs)
               if not doc["content"][idx].get("vl")]
    already = len(pic_idxs) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    if not pending:
        print(f"[SKIP] {in_path.name}（{already} 图均已处理）")
        return

    def _save() -> None:
        in_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")

    def _one(item: tuple[int, tuple[Path, str]]) -> dict:
        idx, (img_path, rel) = item
        itype, idata, u1 = classify_image(client, model, img_path)
        text = ""
        usage = u1
        if idata == "HAS_DATA":
            text, u2 = transcribe_image(client, model, img_path,
                                         _build_query(doc, idx))
            usage = (u1[0] + u2[0], u1[1] + u2[1])
        return {"type": itype, "data": idata, "text": text, "img": rel,
                "_usage": usage}

    failed = 0
    n_data = 0
    tokens = [0, 0]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, item): item[0] for item in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            idx = futs[fut]
            try:
                vl = fut.result()
                usage = vl.pop("_usage")
                tokens[0] += usage[0]
                tokens[1] += usage[1]
                doc["content"][idx]["vl"] = vl
                if vl["data"] == "HAS_DATA":
                    n_data += 1
            except Exception as e:  # noqa: BLE001 —— 单图失败不中断整批，留待续跑
                failed += 1
                print(f"  [FAIL] {in_path.stem} p{doc['content'][idx].get('page')}: {e}")
            if i % 10 == 0 or i == len(pending):
                _save()
    print(
        f"[OK] {in_path.name}: 新处理 {len(pending) - failed}/{len(pending)}"
        f"（其中携带数据 {n_data}），已有 {already}，失败 {failed}；"
        f"tokens 入{tokens[0]}/出{tokens[1]}；{time.perf_counter() - t0:.0f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第 3.5 步：图片多模态序列化（serialized/*.json 原地挂 vl 字段）")
    ap.add_argument("input", type=Path, nargs="?", default=DATA_DIR / "parsed" / "serialized",
                    help="输入目录（默认 data/parsed/serialized，原地更新）")
    ap.add_argument("--raw", type=Path, default=DATA_DIR / "parsed" / "mineru_raw",
                    help="MinerU 原始输出目录（取裁剪图与 img_path）")
    ap.add_argument("--workers", type=int, default=8, help="并发请求数")
    ap.add_argument("--limit", type=int,
                    help="每份文档最多处理的图片数（冒烟测试用）")
    args = ap.parse_args()

    client, model = make_client()
    print(f"# 视觉模型 {model}，输入(=输出) {args.input}，矿 raw {args.raw}，并发 {args.workers}")
    for in_path in sorted(args.input.glob("*.json")):
        process_doc(client, model, in_path, args.raw,
                    workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
