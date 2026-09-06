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

- [ ] 1. Python 项目配置：pyproject / 依赖管理 / 目录结构
- [ ] 2. PDF 解析（原项目用 Docling，届时讨论选型）
- [ ] 3. 解析产物整理与表格序列化
- [ ] 4. 文本切分（原项目用父文档策略）
- [ ] 5. 嵌入与索引
- [ ] 6. 检索（向量检索 + 父文档回溯）
- [ ] 7. LLM 重排序
- [ ] 8. 答案生成（prompt 设计，原项目用结构化输出 + CoT）
- [ ] 9. 端到端串联与冒烟测试

**当前位置：数据集已就绪，准备开始第 2 步（PDF 解析，选型已定为 Docling）。**
（每完成一步，勾选对应项并更新当前位置。）

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
- 机器有 NVIDIA GPU；年报语言为英文 → 解析器跟原项目用 Docling（OCR 配英文）
