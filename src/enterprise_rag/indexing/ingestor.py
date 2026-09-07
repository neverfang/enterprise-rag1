"""嵌入与索引：chunked/*.json -> data/indexes/（FAISS 精确余弦 + BM25，每文档一对文件）。

对应原项目 ingestion.py 的职责，几处刻意保留/调整的设计（2026-09-07 决策讨论）：
- **IndexFlatIP（精确 KNN）而非 ANN / 向量数据库**：归一化后内积 = 余弦，精确搜索
  召回 100%（近似索引只会 ≤）；语料单机离线、按公司路由已由"每文档一索引"实现，
  实验迭代（换嵌入模型/调参）= 换 --out 目录重跑，A/B 目录并存、删目录回滚
- 嵌入抽成 Embedder 类：建库与第 6 步查询端共用同一实例配置，保证同模型同维度
  （train/serve 不一致是检索质量隐形杀手）
- **批量按 token 预算切分**（默认 3,200/请求），而非固定条数——起因：百炼
  qwen3.7-text-embedding-flash 实测有未文档化的单请求 ~4.5K token 上限，超限
  **不报 429 而是挂死**；按预算切批对任何隐藏上限都有防御（已弃用该模型）

产物（data/indexes/，不进 git）：
- {doc}.faiss  —— IndexFlatIP，内部向量序号 == chunk id（按 id 顺序写入）
- {doc}.pkl    —— {"chunk_ids", "bm25": BM25Okapi}，语料序同上
- index_meta.json —— 模型/维度/时间/每文档 chunk 数，供 A/B 目录区分与陈旧检测

嵌入模型：硅基流动 BAAI/bge-m3（免费档：2,000 RPM / 500K TPM；单条文本 8K 上限；
固定 1024 维，不传 dimensions）。TPM 500K => eval 全量（~15M token）建库至少
~30 分钟属正常；dev 集（~1.4M）约 3-5 分钟。

用法：
    python -m enterprise_rag.indexing.ingestor [input_dir] \
        [--out DIR] [--batch 32] [--budget 3200] [--workers 4] [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import faiss
import httpx
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# src/enterprise_rag/indexing/ingestor.py -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

_ENC = tiktoken.get_encoding("o200k_base")
# 单条文本嵌入上限（o200k 计数；bge-m3 官方单条 8192，留 margin）
SINGLE_TEXT_CAP = 8000


def _ntok(text: str) -> int:
    return len(_ENC.encode(text)) if text else 0


def make_embedder() -> "Embedder":
    """从 .env 读取嵌入配置，返回 Embedder（未填 key 直接退出并提示）。"""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("EMBED_API_KEY") or os.environ.get("SILICONFLOW_API_KEY") or ""
    if not api_key or "填" in api_key:
        raise SystemExit(
            "请先在项目根目录 .env 里配置 EMBED_API_KEY"
            "（硅基流动 https://cloud.siliconflow.cn 注册并实名后，API 密钥页创建）")
    base_url = os.environ.get("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
    model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
    dims_env = (os.environ.get("EMBED_DIMS") or "").strip()
    dims = int(dims_env) if dims_env else None  # None: 不传 dimensions，从响应推断
    # - max_retries=0：SDK 内部重试与本类重试叠加会把最坏情况拖到 ~18 分钟，
    #   关掉 SDK 侧，只留 _embed_batch 自己的 3 次退避重试（429 限流也走这里）
    # - trust_env=False：硅基流动是国内域名，绕过系统代理（Windows 上 httpx 会读
    #   注册表代理，Clash 系统代理模式下流量被转一道，多一个故障变量）
    client = OpenAI(
        api_key=api_key, base_url=base_url, max_retries=0,
        http_client=httpx.Client(trust_env=False))
    return Embedder(client, model=model, dims=dims)


class Embedder:
    """可复用嵌入器：建库（embed_corpus）与查询（embed_query）共用。

    批量 = 条数上限 x token 预算双约束（防隐藏的单请求上限，见模块 docstring）；
    单条超预算的文本独占一批不截断（除非超过 SINGLE_TEXT_CAP，仅截嵌入输入，
    BM25 / 重排阶段仍用全文）。
    输出 float32 并 L2 归一化（配合 IndexFlatIP 即余弦）。
    """

    def __init__(self, client: OpenAI, model: str, dims: int | None,
                 batch_size: int = 32, token_budget: int = 3200,
                 retries: int = 3) -> None:
        self.client = client
        self.model = model
        self.dims = dims              # None: 首次响应推断（bge-m3 固定 1024）
        # dimensions 参数只在构造时显式给定维度才发送（bge-m3 不支持该参数，
        # 传了会 400 code 20015；曾因"推断后开始发送"踩坑，故固化此开关）
        self._send_dimensions = dims is not None
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.retries = retries
        self.tokens_in = 0

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """一批 -> (n, dims) float32。按返回 index 排序保证顺序。"""
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                kw: dict = {}
                if self._send_dimensions:  # bge-m3 固定维度，不传；百炼等才传
                    kw["dimensions"] = self.dims
                resp = self.client.embeddings.create(
                    model=self.model, input=texts, timeout=60, **kw)
                if resp.usage:
                    self.tokens_in += resp.usage.prompt_tokens
                rows = sorted(resp.data, key=lambda d: d.index)
                vecs = np.asarray([d.embedding for d in rows], dtype="float32")
                if vecs.shape[0] != len(texts):
                    raise RuntimeError(
                        f"返回条数 {vecs.shape[0]} != 请求条数 {len(texts)}")
                if self.dims is None:
                    self.dims = vecs.shape[1]
                elif vecs.shape[1] != self.dims:
                    raise RuntimeError(f"向量维度 {vecs.shape[1]} != 期望 {self.dims}")
                return vecs
            except Exception as e:  # noqa: BLE001 —— 限流/网络抖动交给重试
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"嵌入失败（已重试 {self.retries} 次）: {last_err}")

    def _pack_batches(self, texts: list[str]) -> list[tuple[int, list[str]]]:
        """切批 -> [(该批在 texts 中的起始下标, 批内容)]。

        条数 <= batch_size 且累计 token <= token_budget；超预算的单条独占一批
        （bge-m3 单条上限 8192，无需为它截断）。
        """
        batches: list[tuple[int, list[str]]] = []
        cur: list[str] = []
        cur_start = 0
        cur_tok = 0
        for i, t in enumerate(texts):
            tk = _ntok(t)
            if cur and (len(cur) >= self.batch_size
                        or cur_tok + tk > self.token_budget):
                batches.append((cur_start, cur))
                cur, cur_start, cur_tok = [], i, 0
            cur.append(t)
            cur_tok += tk
        if cur:
            batches.append((cur_start, cur))
        return batches

    def embed_corpus(self, texts: list[str], workers: int = 4) -> np.ndarray:
        """全量文本 -> (n, dims) 已归一化。批次并发，结果按起始下标回填原顺序。"""
        for t in texts:
            if not t or not t.strip():
                raise ValueError("遇到空文本，chunked 产物不应有空块，请先检查切分")
        for t in texts:
            if _ntok(t) > SINGLE_TEXT_CAP:  # 仅嵌入截断；BM25/重排仍用全文
                print(f"  [WARN] 单块 {_ntok(t)} tok 超单条上限，嵌入仅取前 "
                      f"{SINGLE_TEXT_CAP} tok（BM25/重排不受影响）", flush=True)
        batches = self._pack_batches([
            _ENC.decode(_ENC.encode(t)[:SINGLE_TEXT_CAP]) if _ntok(t) > SINGLE_TEXT_CAP else t
            for t in texts])
        if self.dims is None:
            # 首批先行（串行）：推断维度并分配输出数组
            start, b = batches[0]
            v = self._embed_batch(b)
            self.dims = v.shape[1]
            out = np.zeros((len(texts), self.dims), dtype="float32")
            out[start:start + len(b)] = v
            batches = batches[1:]
        else:
            out = np.zeros((len(texts), self.dims), dtype="float32")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self._embed_batch, b): (start, len(b))
                    for start, b in batches}
            for fut in as_completed(futs):
                start, n = futs[fut]
                vecs = fut.result()  # 失败直接抛，整份文档重建（见 skip 逻辑）
                out[start:start + n] = vecs
        faiss.normalize_L2(out)
        return out

    def embed_query(self, text: str) -> np.ndarray:
        """单条查询 -> (dims,) 已归一化。"""
        vec = self._embed_batch([text])[0]
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec


def bm25_tokenize(text: str) -> list[str]:
    """英文友好分词：小写；数字保留千分位/小数点（"86.6"、"1,234"），单词 >=2 字符。"""
    return re.findall(r"\d+(?:[.,]\d+)*|[a-z][a-z0-9]*", text.lower())


def build_doc(doc: dict, out_dir: Path, embedder: Embedder,
              workers: int = 4) -> dict:
    """一份 chunked 文档 -> {doc_id}.faiss + {doc_id}.pkl，返回统计信息。"""
    chunks = sorted(doc.get("chunks", []), key=lambda c: c["id"])
    ids = [c["id"] for c in chunks]
    if ids != list(range(len(chunks))):
        raise RuntimeError(f"chunk id 不连续: 期望 0..{len(chunks) - 1}")
    texts = [c["text"] for c in chunks]

    # 向量索引：IndexFlatIP（精确余弦），按 chunk id 顺序写入 -> 内部序号 == chunk id
    t0 = time.perf_counter()
    vecs = embedder.embed_corpus(texts, workers=workers)
    if vecs.shape[1] != embedder.dims:
        raise RuntimeError(f"向量列数 {vecs.shape[1]} != 推断维度 {embedder.dims}")
    index = faiss.IndexFlatIP(embedder.dims)
    index.add(vecs)
    stem = (doc.get("metainfo") or {}).get("doc_id") or "doc"
    faiss.write_index(index, str(out_dir / f"{stem}.faiss"))

    # BM25 索引：同一批文本、同一顺序，检索端可按 chunk id 对齐两路分数
    bm25 = BM25Okapi([bm25_tokenize(t) for t in texts])
    with open(out_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump({"chunk_ids": ids, "bm25": bm25}, f)

    return {
        "stem": stem,
        "n_chunks": len(chunks),
        "tokens_in": embedder.tokens_in,
        "secs": time.perf_counter() - t0,
        "sha1": (doc.get("metainfo") or {}).get("sha1"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="第 5 步：嵌入与索引（chunked -> data/indexes，FAISS 精确余弦 + BM25）")
    ap.add_argument("input", type=Path, nargs="?", default=DATA_DIR / "parsed" / "chunked",
                    help="输入目录（默认 data/parsed/chunked）")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "indexes")
    ap.add_argument("--batch", type=int, default=32, help="单请求最大条数（硅基流动 32）")
    ap.add_argument("--budget", type=int, default=3200,
                    help="单请求 token 预算（o200k 计数，防隐藏单请求上限）")
    ap.add_argument("--workers", type=int, default=4, help="并发请求数")
    ap.add_argument("--limit", type=int, help="最多处理 N 份文档（冒烟用）")
    ap.add_argument("--force", action="store_true", help="已有索引也重建")
    args = ap.parse_args()

    embedder = make_embedder()
    embedder.batch_size = args.batch
    embedder.token_budget = args.budget
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input.glob("*.json"))
    if args.limit is not None:
        paths = paths[:args.limit]
    print(f"# 嵌入模型 {embedder.model}，输入 {args.input}，输出 {args.out}，"
          f"批量 {args.batch} 条 / {args.budget} tok x {args.workers} 并发", flush=True)

    meta_docs: dict[str, dict] = {}
    t_all = time.perf_counter()
    done = skipped = failed = 0
    for in_path in paths:
        stem = in_path.stem
        if not args.force and (args.out / f"{stem}.faiss").exists() \
                and (args.out / f"{stem}.pkl").exists():
            print(f"[SKIP] {in_path.name}（索引已存在）", flush=True)
            skipped += 1
            continue
        try:
            doc = json.loads(in_path.read_text(encoding="utf-8"))
            st = build_doc(doc, args.out, embedder, workers=args.workers)
        except Exception as e:  # noqa: BLE001 —— 单文档失败不中断整批，留待续跑
            failed += 1
            print(f"[FAIL] {in_path.name}: {e}", flush=True)
            continue
        done += 1
        meta_docs[st["stem"]] = {
            "n_chunks": st["n_chunks"], "sha1": st["sha1"],
            "source": in_path.name,
        }
        print(f"[OK] {in_path.name}: {st['n_chunks']} chunks -> "
              f"{st['stem']}.faiss/.pkl，tokens 入{st['tokens_in']}"
              f"（免费），{st['secs']:.0f}s", flush=True)

    if meta_docs:
        meta_path = args.out / "index_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) \
            if meta_path.exists() else {}
        meta.update({
            "model": embedder.model, "dims": embedder.dims,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "docs": {**meta.get("docs", {}), **meta_docs},
        })
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"# 完成 {done}（跳过 {skipped}，失败 {failed}），"
          f"总 tokens 入{embedder.tokens_in}，总耗时 {time.perf_counter() - t_all:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
