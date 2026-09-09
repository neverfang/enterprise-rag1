# Enterprise RAG — 企业年报问答系统

对上百页的英文企业年报 PDF 做问答：给出问题，系统路由到对应公司、检索证据页、
返回结构化答案（数值/真假/人名/名单）与页级引用。参考
[IlyaRice/RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2)（ERC2 获奖方案）
的流水线逐步自行实现，部分环节做了自己的改进（见下）。

**状态**（2026-09-08）：9 个阶段全部实现并在 dev 集（10 份年报）验证通过；
eval 全量（100 份）数据处理与正式评测未跑——续跑入口见[局限与后续](#局限与后续)。

## 流水线

```
PDF ──► ⓪ 文本层质量预检 ──► ① MinerU 解析 ──► ② 表格 LLM 序列化 ──► ③ 图表多模态转写
              │                  │                    │
              └──────────────────┴────────────────────┘
                                 ▼
              ④ 结构化切分（标题硬边界 + 面包屑 + els 区间）
                                 │
                                 ▼ 可选，默认关闭
              ④.5 正文 Chunk LLM 清洗与上下文增强
                                 ▼
              ⑤ 索引（bge-m3 × FAISS 精确余弦 + BM25，每文档一索引）
                                 ▼
        问题 ──► ⑥ 路由 + 双路召回融合 top-20 ──► ⑦ LLM 重排 top-6
                                 ▼
              ⑧ 题型化 schema 生成（CoT + 页引用 + 不编造）
                                 ▼
                    ⑨ 批量串联 + 评测（比较题三步）
```

| # | 阶段 | 模块 | 技术选型 | dev 实测 |
|---|------|------|----------|----------|
| 0 | PDF 质量预检 | `quality/quality_checker.py` | PyMuPDF 前 5 页；有效字符率阈值 80% + 文本密度 | 入口默认开启 |
| 1 | PDF 解析 | `parsing/pdf_parser.py` | MinerU pipeline 后端（GPU） | 10/10，~293s/份，1164 页 873 表 |
| 2 | 表格序列化 | `processing/table_serializer.py` | DeepSeek，json_object + pydantic 校验 | 873 表 → 6,262 个上下文独立信息块，¥5 / 8 分钟 |
| 3 | 图片序列化 | `processing/image_serializer.py` | deepseek-v4-flash-vision-exp 两阶段（先分类后转写） | 197 图，41 张含数据的完成转写，<¥1 |
| 4 | 文本切分 | `processing/text_splitter.py` | 自实现：标题硬边界 + 段落贪心 ≤300 tok + 标题面包屑 | 4,694 chunks，p95=288 tok |
| 4.5 | 正文清洗（可选） | `processing/chunk_transformer.py` | LLM 相邻上下文去噪 + 数字守恒校验 + 失败回退 | 默认关闭，按调用量计费 |
| 5 | 嵌入索引 | `indexing/ingestor.py` | 硅基流动 bge-m3（免费）+ FAISS IndexFlatIP + BM25Okapi | 1.24M token，18 秒，免费 |
| 6 | 检索 | `retrieval/retriever.py` | 正则路由 + 向量/BM25 双路 min-max 融合 top-20 | dev 5 题路由全对 |
| 7 | LLM 重排 | `reranking/reranker.py` | DeepSeek 0-1 锚点量表，combined = 0.7×LLM + 0.3×融合分 | batch=10，~¥0.01-0.02/题 |
| 8 | 答案生成 | `generation/generator.py` | 四题型 schema + CoT + relevant_pages 校验 + 弱证据提示 | ~¥0.02-0.05/题 |
| 9 | 串联评测 | `pipeline.py` + `evaluate.py` | 断点续跑、比较题三步、自用评分 | dev 5 题 + round2 40 题 0 错误 |

## 设计决策

**忠实获奖方案的部分**（读源码核实的配方，非道听途说）：

- **表格序列化**：每张表用 LLM 改写成若干"上下文独立的信息块"（自带主语、年份、
  单位），让表格可被嵌入检索——这是原方案对表格问答的核心解法
- **路由**：公司名做正则词边界匹配，命中即删继续找，天然支持多公司比较题
- **重排**：0-1 锚点量表逐块打分（每档有文字定义），与检索分 0.7/0.3 加权；
  只用检索分排序会在"要算 margin 还是直接读 margin"这类题上吃亏
- **生成**：按题型（number/boolean/name/names）分 schema；number 型严格规则
  （单位换算、括号负数、币种不符→N/A、拒绝自行计算推导）；引用页只允许来自
  实际提供的上下文；比较题拆成每家子问题再汇总
- **45/100 评测题故意不可答**：系统必须敢于回 N/A，乱答双倍扣分

**超出原方案的部分**（dev 集上有验证依据）：

- **入口质量门禁**：MinerU 前快速抽取前 5 页文本；有效字符率低于 80% 或完全
  没有可识别文本时拒绝入库，同时返回可供 Dashboard 使用的结构化状态和提示
  “该文档质量不达标，请检查后重新上传”。字符密度和文本页比例用于诊断，避免
  仅凭稀疏封面误拒正常年报
- **双路召回融合**：原版获奖流水线其实只走向量 top-28、BM25 建了没用（读源码
  纠正的误记）。我们双路各取 top-30，min-max 归一化后 vw=0.6 融合——dev 抽查
  两路 top3 重叠仅 0-1/3（向量补同义词、BM25 补精确数字），互补性实证
- **结构化切分**：原版用 langchain 递归切 300/50。我们以标题为硬边界、段落贪心
  合并、每块带标题面包屑前缀（序列化思想的正文版，零 LLM 成本）；chunk 元数据
  记录内容流元素区间，父文档可用跨页元素窗口而非页边界构建
- **父文档预算制**：重排后回溯整页/元素窗口给生成端，els_window 模式超 60K
  字符预算时窗口减半重建而非丢段
- **图表转写禁估值**：只允许转写图上印出的数字，禁止按网格线估读（实测模型
  会"估"出似是而非的值）
- **弱证据提示**：重排分 ≤0.5 时提示模型"认真考虑 N/A"——dev 集上可答题 top 分
  0.9-1.0、弱证据题 ≤0.7，信号可用；但只作参考不作门槛

## 结果

- dev 测试集 5 题（含数值、布尔、跨页证据题）：路由 5/5，答案与单步 CLI 一致；
  父文档回溯救回过一题 chunk 级检索平淡的收购题（整页上下文含叙述）
- round2 40 题：7 题可路由（含 1 道比较题三步走通）、33 题路由失败按设计回 N/A
  （当时只建了 10 份索引；eval 全量建索引后不应发生）
- **基线参照**：获奖 o3-mini 提交在我们自用评分规则下 **81.0%**（number 86.2 /
  boolean 87.5 / name 77.8 / names 33.3 / N/A 题 91.1 / 非 N/A 72.7）——
  正式评测时可对照的及格线
- 自用评分规则：number 相对误差 ≤1%、boolean 真值、name 规范化后相等/互含、
  names 集合严格相等、N/A 题单独统计（见 `evaluate.py` 文档串）

## 快速开始

环境：Windows + NVIDIA GPU（解析用；CPU 也能跑只是慢）、Python 3.12（uv 管理）。

```powershell
# 1) 环境（国内镜像 + modelscope 模型源）
uv venv --python 3.12 .venv
uv pip install -p .venv\Scripts\python.exe -e .
# GPU 加速：覆盖 CUDA 版 torch（MinerU 检测到 CUDA 自动用 GPU）
uv pip install -p .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# 2) .env（不进 git；键如下，值见各服务商）
#    DEEPSEEK_API_KEY / LLM_BASE_URL / LLM_MODEL     序列化、重排、生成
#    LLM_VISION_MODEL                                  图片转写（key 复用上行）
#    EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL      嵌入（硅基流动 bge-m3，免费）
#    DASHSCOPE_API_KEY                                 可选：百炼嵌入 A/B

# 3) 数据准备：data/raw/dev/ 放 PDF（data/pdf_metadata.csv 有 sha1↔公司名映射）
$env:MINERU_MODEL_SOURCE='modelscope'
$py = .venv\Scripts\python.exe
$py -m enterprise_rag.parsing.pdf_parser data\raw\dev        # ⓪ 预检 + ① 解析（断点续跑）
# 仅在人工确认必须绕过门禁时使用：末尾加 --no-quality-check
$py -m enterprise_rag.processing.table_serializer            # ② 表序列化
$py -m enterprise_rag.processing.image_serializer            # ③ 图序列化
$py -m enterprise_rag.processing.text_splitter               # ④ 切分
# 可选：切分后立即清洗正文 Chunk（默认关闭，会产生 LLM 费用）
$py -m enterprise_rag.processing.text_splitter --transform
# 或对已有 chunked 产物独立、可续跑地执行
$py -m enterprise_rag.processing.chunk_transformer
$py -m enterprise_rag.indexing.ingestor                      # ⑤ 建索引

# 4) 问答与评测
$py -m enterprise_rag.pipeline data\questions_test_set.json --out data\answers_run_dev.json
$py -m enterprise_rag.evaluate data\answers_run_dev.json     # 对 answers_eval 评分+基线对比
```

各阶段都支持断点续跑（已完成的自动跳过）；单阶段 CLI 冒烟入口见各模块文档串。

## 成本（dev 10 份实测）

| 阶段 | 耗时 | 费用 |
|------|------|------|
| 解析（GPU） | ~49 分钟 | 0 |
| 表格序列化 | ~8 分钟 | ~¥5（2.1M tokens） |
| 图片序列化 | ~2 分钟 | <¥1 |
| 建索引 | 18 秒 | 0（bge-m3 免费档） |
| 问答全链路 | — | ~¥0.03-0.07/题（100 题约 ¥3-7） |

eval 100 份外推：解析 ~8 小时（GPU）、序列化 ~¥50、索引 ~30 分钟。

## 局限与后续

- **eval 全量未跑**：本机内存紧张，解析 5/100 后暂停。续跑入口：
  `bash run_eval_prep.sh`（链式五阶段、全程断点续跑），完成后
  `python -m enterprise_rag.pipeline data/questions_eval.json --out data/answers_run_eval.json`
- names 类题（多元素名单）是已知弱项——基线也只有 33.3%，严格集合相等偏严
- 路由依赖公司名出现在题面；未命中直接回 N/A（eval 100 份全建索引后，
  官方题均含公司名，风险低）
- 解析瑕疵已审计（2026-09-08）：无边框表"两数值挤一格"经抽查由序列化 LLM
  正确按年度归属，无需修复；跨页表经严格判据核查无真断表（MinerU 自行合并），
  未引入合并逻辑

## 项目结构

```
src/enterprise_rag/
├── quality/quality_checker.py     # ⓪ PDF 文本层质量预检 + 结构化指标
├── parsing/pdf_parser.py        # ① MinerU 解析 → data/parsed/docs
├── processing/                  # ②③④/④.5 序列化/转写/切分/可选清洗
│   ├── normalize.py             #    文本规范化（连字断痕、控制字符）
│   ├── table_serializer.py      #    → data/parsed/serialized
│   ├── image_serializer.py      #    图片分类 + 含数据图转写
│   ├── text_splitter.py         #    → data/parsed/chunked
│   └── chunk_transformer.py     #    可选正文 LLM 清洗 → data/parsed/transformed
├── indexing/ingestor.py         # ⑤ 嵌入 + FAISS + BM25 → data/indexes
├── retrieval/retriever.py       # ⑥ 路由 + 双路融合 + 父文档回溯
├── reranking/reranker.py        # ⑦ LLM 重排
├── generation/generator.py      # ⑧ 题型化生成
├── pipeline.py                  # ⑨ 批量问答（断点续跑、比较题三步）
└── evaluate.py                  #    评分 + 基线对比
```

## License

MIT
