# Enterprise RAG

基于企业年报 PDF 的问答 RAG 系统。参考 [IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2)（Enterprise RAG Challenge 2 获奖方案）自行实现。

## 流水线

```
PDF 解析(Docling) → 结构整理/表格序列化 → 文本切分(父文档策略)
→ 嵌入 + 索引 → 向量检索 + 父文档回溯 → LLM 重排序 → 结构化答案生成
```

## 快速开始

```bash
# 1. 安装（建议 Python 3.10+）
pip install -e ".[dev]"

# 2. 配置 API Key
cp .env.example .env   # 填入 OPENAI_API_KEY（可选 OPENAI_BASE_URL）

# 3. 放置测试数据：data/test_set/ 下放 PDF 和 questions.jsonl

# 4. 分阶段运行
python main.py parse   # 解析 PDF → data/parsed/
python main.py ingest  # 嵌入 + 建索引
python main.py ask "去年的营收是多少"  # 端到端问答

# 或一键冒烟测试
python scripts/smoke_test.py
```

## 项目结构

```
configs/                     # 命名实验配置（default.yaml）
data/
  raw/                       # 原始 PDF（不进 git）
  parsed/                    # 各阶段中间产物（不进 git）
  test_set/                  # 小样例数据，用于冒烟测试
src/enterprise_rag/
  parsing/                   # PDF → 结构化文档
  chunking/                  # 切分（子块 + 父文档）
  indexing/                  # 嵌入与向量索引
  retrieval/                 # 检索 + 父文档回溯
  reranking/                 # LLM 重排序
  generation/                # 答案生成、prompt 模板
  llm/                       # LLM/Embedding API 封装
  pipeline.py                # 端到端编排
main.py                      # CLI 入口
scripts/                     # 冒烟测试等脚本
tests/                       # pytest 单测
```

## License

MIT
