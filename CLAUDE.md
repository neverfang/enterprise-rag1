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

- `data/raw/`：10 份英文年报 PDF（来自 ERC2 比赛语料，sha1 命名，不进 git）
  - 5 份来自 IlyaRice 测试集 + 5 份来自官方 trustbit/enterprise-rag-challenge 的 round2/samples
- `data/questions_test_set.json`：5 个问题（对应 IlyaRice 测试集 5 份 PDF，无标准答案）
- `data/questions_round2.json`：40 个问题（对应官方 samples 20 份 PDF，含 schema 类型）
- 机器有 NVIDIA GPU；年报语言为英文 → 解析器跟原项目用 Docling（OCR 配英文）
