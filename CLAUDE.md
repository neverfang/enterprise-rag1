# CLAUDE.md

企业年报问答 RAG 系统，参考 IlyaRice/RAG-Challenge-2（ERC2 获奖方案）自行实现。
用户的工作语言是中文，回复、注释、commit message 说明用中文（commit 标题可用英文）。

## Git 工作流（用户已授权）

- 远程：origin = https://github.com/neverfang/enterprise-rag1.git，主分支 main
- 每完成一个完整的功能单元：commit 并 push 到 origin main，不必逐文件碎提交
- 不要提交 .env、data/raw/、data/parsed/ 的内容

## 流水线阶段（每阶段产物落盘，下游可独立重跑）

```
parse        PDF → 结构化文档（Docling）→ data/parsed/docs/
serialize    表格 → 文本/Markdown
chunk        切分子块 + 父文档
ingest       嵌入 + 建索引 → data/parsed/index/
retrieve     向量检索 top-k + 父文档回溯
rerank       LLM 重排序，取 top-n
answer       结构化输出 + CoT 生成答案
```

## 技术选型（已定，勿随意更换）

- Python 3.10+；`pip install -e ".[dev]"`
- PDF 解析：Docling
- LLM / Embedding：OpenAI 兼容 API，从 .env 读 OPENAI_API_KEY / OPENAI_BASE_URL / 模型名写在 configs/*.yaml
- 向量索引：先用 numpy 本地余弦相似度（简单可控），接口留成可替换，后续可换 Qdrant
- CLI：typer，入口 main.py；测试 pytest

## 约定

- 一切参数走 configs/*.yaml + .env，禁止硬编码模型名、top-k、chunk 大小
- 所有 prompt 集中在 src/enterprise_rag/generation/prompts.py，不散落
- 阶段间通过 data/parsed/ 下的文件交换数据（JSON/JSONL/npz），格式变更要向后兼容或写迁移
- 纯逻辑模块（chunking、retrieval）优先写单测；改完跑 `python scripts/smoke_test.py`
- 代码注释密度保持与现有文件一致，中文注释
