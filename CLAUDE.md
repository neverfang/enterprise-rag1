# CLAUDE.md

企业年报问答 RAG 系统，参考 IlyaRice/RAG-Challenge-2（ERC2 获奖方案）逐步实现。

## 工作方式（重要）

- 用户要**一步一步**生成代码：每次只实现当前步骤，禁止提前创建后续模块的文件、
  存根、"顺带"实现还没讨论到的功能
- 每步动手前，先说明这一步做什么、涉及哪些文件，等用户确认再写
- 交流、注释、commit 说明用中文（commit 标题可用英文）

## Git（用户已授权）

- origin = https://github.com/neverfang/enterprise-rag1.git，分支 main
- 每完成一个步骤：commit 并 push
- 不提交 .env 和数据文件

## 开发路线图（参考原项目流水线，按序推进）

- [x] 1. Python 项目配置：pyproject / 依赖管理 / 目录结构（uv + venv 3.12）
- [x] 2. PDF 解析（MinerU pipeline，封装在 src/enterprise_rag/parsing/pdf_parser.py）
- [ ] 3. 解析产物整理与表格序列化
- [ ] 4. 文本切分（原项目用父文档策略）
- [ ] 5. 嵌入与索引
- [ ] 6. 检索（向量检索 + 父文档回溯）
- [ ] 7. LLM 重排序
- [ ] 8. 答案生成（prompt 设计，原项目用结构化输出 + CoT）
- [ ] 9. 端到端串联与冒烟测试

**当前位置：第 2 步完成（dev 集 10 份已解析），准备开始第 3 步（解析产物整理与表格序列化）。**
（每完成一步，勾选对应项并更新当前位置。）

## 解析阶段结果（第 2 步完成，2026-09-06）

- dev 集 10/10 解析成功（GPU，平均 ~293s/份）：共 1164 页、873 个表格
- 产物：`data/parsed/docs/*.json`（统一格式：metainfo+公司名注入 / content 顺序流 / tables）；
  MinerU 原始输出保留在 `data/parsed/mineru_raw/`（layout.pdf / span.pdf 可视化可人工抽查）
- 质量：无文档级乱码（异常字符率 0.03%–0.13%，主体是换行符和零星控制符）；
  页眉页脚由 MinerU 标记 discarded、转换层过滤（每份 152–464 条）；dev 集未出现旋转表格
- 已知瑕疵（留给第 3 步处理）：
  - 4/873 个表格结构退化，均为**无边框表**（列间仅空白分隔，两个年度数值被挤进同一格；
    数值未丢失，LLM 仍可读，线性化时注意）
  - 文本流含少量控制字符（如 CrossFirst 的 U+0003/U+0004 共 29 个），第 3 步规范化时清除
- 批量入口：`.venv\Scripts\python.exe -m enterprise_rag.parsing.pdf_parser data\raw\dev`
  （已解析的自动跳过，可断点续跑；eval 100 份用同命令换目录即可）

## 数据集（已就绪，2026-09-06）

- `data/raw/dev/`：10 份英文年报 PDF，开发/冒烟集（不进 git），已按公司名重命名
  - 5 份来自 IlyaRice 测试集（Holley、Tradition、TSX_Y、Mercia、CrossFirst）
  - 5 份来自官方 trustbit round2/samples（Global Medical REIT、Zegona、TD SYNNEX、Air Products、Sandwell Aquatics）
- `data/raw/eval/`：100 份英文年报 PDF（504 MB），官方 round2 最终评测语料（不进 git）
- `data/pdf_metadata.csv`：110 行 sha1 ↔ 公司名 ↔ 文件名映射（filename 相对 data/raw/，含 split 来源/币种/行业），解析时把公司名注入元信息用
- `data/questions_test_set.json`：5 题（dev 用，无标准答案）
- `data/questions_round2.json`：40 题（dev 的 round2 samples 部分用，无标准答案）
- `data/questions_eval.json`：100 题（评测集，官方最终题）
- `data/answers_eval.json`：100 条人工校对标准答案（49 条带 reference_pools 出处池；45 条答案含 N/A——故意的不可答题，考系统不编造）
- `data/answers_baseline_o3mini.json`：获奖系统提交（100 条，带 references 和完整 reasoning_process，可作对比基线）
- 机器有 NVIDIA GPU（RTX 4060 Laptop 8GB）；年报语言为英文
- **解析器选型：MinerU pipeline 后端**（用户选定；原项目用 Docling，想对比时评测集现成）
- 环境：`.venv` = Python 3.12（uv 管理；MinerU 在 Windows 要求 3.10–3.12），
  安装用清华镜像，模型下载走 modelscope（MINERU_MODEL_SOURCE=modelscope）
- **GPU 加速**：mineru[core] 默认装 CPU 版 torch；装完依赖后需再执行
  `uv pip install -p .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu126 torch torchvision`
  覆盖为 CUDA 版，MinerU 检测到 CUDA 后自动用 GPU（RTX 4060 8GB）
