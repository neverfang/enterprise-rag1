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
- [x] 3. 解析产物整理与表格序列化（src/enterprise_rag/processing/，DeepSeek）
- [x] 3.5 图片多模态序列化（deepseek-v4-flash-vision-exp，src/enterprise_rag/processing/image_serializer.py）
- [x] 4. 文本切分（src/enterprise_rag/processing/text_splitter.py，chunk + 页父文本 + els 区间）
- [x] 5. 嵌入与索引（src/enterprise_rag/indexing/ingestor.py，硅基流动 bge-m3 + FAISS + BM25）
- [x] 6. 检索（src/enterprise_rag/retrieval/retriever.py，路由 + 向量/BM25 双路融合 + 父文档回溯）
- [x] 7. LLM 重排序（src/enterprise_rag/reranking/reranker.py，DeepSeek 逐块打分 + combined 0.7/0.3）
- [x] 8. 答案生成（src/enterprise_rag/generation/generator.py，题型化 schema + CoT + 弱证据提示）
- [x] 9. 端到端串联与冒烟测试（src/enterprise_rag/pipeline.py + evaluate.py，dev 验证通过）

**当前位置：第 9 步 dev 验证完成（pipeline + 评测脚本就绪，基线参照分 81.0%）。下一步：eval 100 份全量数据处理（解析 ~8h GPU -> 序列化 -> 建索引），再正式跑 100 题评测。**
（每完成一步，勾选对应项并更新当前位置。）

## 端到端阶段结果（第 9 步 dev 验证完成，2026-09-08）

- `src/enterprise_rag/pipeline.py`：批量问答入口。题目字段自适应
  （text|question、kind|schema）；**断点续跑**（输出里无 error 的题跳过，
  每答完一题即落盘，中断可续）；路由 0 家 -> 快速 N/A（dev 部分建索引时
  跳过未索引公司，eval 全量后不应发生）；输出含 references
  （pdf_sha1 + page_index，sha1 来自 pdf_metadata.csv）
- **比较题三步**（忠实原版 process_comparative_question 流程，prompt 自写）：
  LLM 改写成每家子问题 -> 各家并行 number 型作答 -> LLM 汇总答原题。
  冒烟：GMREIT vs Zegona D/E 题三步走通，双 N/A（比率需自行计算，number
  schema"拒绝推导"的正确行为）；已知边界：另一家未索引的比较题路由只剩
  1 家、用原题单家检索（eval 100 份全索引后不存在此情况）
- `src/enterprise_rag/evaluate.py`：自用评分（开发迭代反馈，非官方口径）——
  number 相对误差 <=1%、boolean 真值、name 规范化后相等/互含、names 集合
  严格相等（偏严，但双方同一把尺）；gt 的 answers 字段引号风格不一
  （直/弯/混合），解析带降级链；N/A 题单独统计
- **基线参照分**（获奖 o3-mini 提交，我们的规则）：**总体 81.0%**
  （number 86.2 / boolean 87.5 / name 77.8 / names 33.3 /
  N/A 题 91.1 / 非 N/A 72.7）
- dev 验证：test_set 5 题（与第 8 步 CLI 答案一致，0 错误）；round2 40 题
  （7 题可路由含 1 道比较题、33 题路由失败走快速 N/A、输入含 1 道重复题
  -> 输出 39 条属预期）
- 冒烟踩的坑：①boolean schema 只收 bool 与弱证据提示（"回 N/A"）自相
  矛盾，模型回 'N/A' 字符串导致重试 3 次全挂 -> `bool | Literal["N/A"]`；
  ②续跑最初把 error 行当已答 -> 只收无 error 行
- 运行产物 answers_run*.json 不进 git（.gitignore）
- 复跑：`.venv\Scripts\python.exe -m enterprise_rag.pipeline <questions.json>
  --out <answers_run.json> [--limit N --workers 2]`；评测
  `python -m enterprise_rag.evaluate <answers_run.json>`

## 生成阶段结果（第 8 步完成，2026-09-08）

- 模块 `src/enterprise_rag/generation/generator.py`：`Generator.answer()`——
  父文档上下文 + 题型化提示词 -> `{step_by_step_analysis, reasoning_summary,
  relevant_pages, final_answer}`；CLI 串全链路（路由->融合->重排->父文档->生成）
- **题型化 schema**（题目 json 的 `kind` 字段路由，eval 100 题分布
  number 58 / boolean 24 / names 9 / name 9）：忠实原版 4 套
  AnswerWithRAGContext*Prompt——number 的严格规则全保留（单位
  thousands/millions 换算、括号=负数、币种不符->N/A、**拒绝计算推导**、
  度量概念必须精确匹配）；上下文格式 `Text retrieved from page {N}`
  是 relevant_pages 的页号锚点，同原版
- **relevant_pages 校验**（同原版 _validate_page_references）：只保留
  上下文实际提供的页（防幻觉页码），不足 2 页按重排序补足、至多 8 页
- **弱证据提示注入**（我们的，原版没有）：第 7 步发现可答题 top 重排分
  0.9-1.0、弱证据题 <=0.7——top 分 <=0.5 时 system 追加"证据可能不足，
  认真考虑 N/A"。**是参考不是门槛**（Mercia 0.6 未触发、照常作答且对）
- 上下文 60K 字符保险丝（超限丢靠后的页）；CLI `--kind auto` 简易猜题
  （正式批量读 json 的 kind）
- 冒烟（dev 5 题，全链路）：
  - Tradition(number) -> **9.9**：p45 表里三个 margin 变体（adjusted
    underlying 12.7 / adjusted 11.4 / reported 9.9）中按"题目无限定词取
    reported"正确选中；引用页 [45, 74] 真实
  - 三个 boolean -> True：TSX_Y（NCIB 2021-08 宣布 + Plan of Arrangement
    2022-08 董事会批准）；Holley（5 页收购明细，CoT 逐一列举 Baer/Simpson/
    Drake 等）；**Mercia 父文档救回**——chunk 级检索平淡，但整页上下文含
    "acquisition of the VCT fund management business"（p42/p78），
    之前"疑似不可答"判断被推翻，父文档回溯价值实证
  - CrossFirst(number) -> **N/A**：top 分 0.4 触发弱证据提示，正确识别
    "全员工薪酬 ≠ 高管薪酬（SCT 在 proxy statement）"；风险：高管表
    曾在融合 #6，重排未进 top6——真伪留给第 9 步评测
- 成本：全链路（重排+生成）~13-32K 入 tokens/题（约 ¥0.02-0.05），
  100 题 ¥2-5
- 复跑：`.venv\Scripts\python.exe -m enterprise_rag.generation.generator "问题"`
  （--kind auto|name|names|number|boolean --top-n 6 --weak-th 0.5）

## 重排阶段结果（第 7 步完成，2026-09-08）

- 模块 `src/enterprise_rag/reranking/reranker.py`：`Reranker`（融合 top-20 →
  DeepSeek 逐块相关性打分 → combined 排序取 top-6）；CLI 串联 Router +
  Retriever + Reranker 全链路
- **忠实原版的配方**：0-1 锚点量表 prompt（0.1 步进、每档文字定义）、
  `Block N` 多块格式 + "exactly N rankings, in order"、
  combined = 0.7×LLM + 0.3×检索分、缺分兜底 0.0。两处替换：模型
  deepseek-chat（json_object + pydantic 校验，同 table_serializer 模式）；
  加权项用第 6 步**融合分**替代原版的向量 distance（方向一致且含双路信息）
- **批量 10 块/prompt**（原版调用处 2 块/prompt）：同题对比 top-3 成员一致，
  批内有对照打分反而更稳（明写 margin 的表 1.0 vs 待计算表 0.9；batch=2
  孤立打分同表只给 0.4），tokens 7.9K vs 12.6K、请求数 2 vs 10（系统提示
  不再重复 10 次）——批内互相干扰的担忧未成立
- 每块截断 cap=1,500 tok（表块最大 5.7K）：只影响重排输入，第 8 步生成
  仍用完整块/父文档；数量不符/JSON 越界重试 3 次，耗尽才兜底 0 分（冒烟
  5 题 × ~30 块零兜底触发）
- 冒烟（dev 5 题）：
  - Tradition：p45 利润调节表 #2→#1（表里**明写 margin 9.9%/8.4%**，比要
    自己算的收入表更直接）
  - TSX_Y：融合 **#20** 的股东信 → 重排 #1（$100M repurchase/NCIB）——
    top-20 全量送重排的价值证明；Plan of Arrangement 细节块被降权
    （描述过往交易 ≠ 宣布计划）
  - Holley：融合 #18 goodwill 变动表 → #3（llm=1.0）、#4 Acquisitions
    章节 → #1；Mercia/CrossFirst 最高 LLM 分仅 0.7/0.4
  - **可答性信号**：可答题 top LLM 分 0.9-1.0，弱证据题 ≤0.7——第 8 步
    N/A 判断可用
- 成本：~7-20K 入 / ~1K 出 tokens/题（约 ¥0.01-0.02），100 题约 ¥1-2
- 复跑：`.venv\Scripts\python.exe -m enterprise_rag.reranking.reranker "问题"`
  （--batch 10 --llm-w 0.7 --top-n 6 --cap 1500；--batch 2 切回原版粒度对比）

## 检索阶段结果（第 6 步完成，2026-09-07）

- 模块 `src/enterprise_rag/retrieval/retriever.py`：`Router`（问题→doc_id）+
  `Retriever`（单文档双路召回+融合、`parent_context` 父文档回溯）
- **路由**：忠实原版正则法——公司名 ∪ 文件名 stem 做词边界匹配
  （`re.escape(key) + (?:\W|$)`，长键优先），命中即从问题文本删除继续找，
  天然支持比较题多公司；只路由已建索引的文档。dev 5 题 + 比较场景路由全对
- **双路融合（方案 B，超出原版）**：原版获奖流水线其实**只走向量 top-28，
  BM25 建了没用**（读源码纠正了此前的误记）；我们召回 = FAISS top-k +
  BM25 top-k（默认 k=30），各路分数在自身 top-k 内 min-max 归一化后按
  vw=0.6 加权求和（某路没进 top-k 记 0，偏向两路都命中的块），融合出
  top-20 候选——是第 7 步 LLM 重排的输入。升级依据：dev 5 题抽查两路
  top3 重叠仅 0-1/3（向量补同义词 buyback→repurchase、BM25 补精确词/数字）
- **父文档回溯 `parent_context`**：两模式可切换——
  - `pages`（同原版）：chunk 所在页整页文本，按页去重
  - `els_window`（第 4 步钩子兑现）：chunk 的 els 元素区间两侧各扩 W 个
    元素（默认 15）从内容流拼文本（标题#/表序列化块/vl 转写），相邻区间
    合并——跨页的语义连续段落不被页边界切断
  - 冒烟观察：W=15 时 6 块 → 4 段 90K 字符，比 pages（33K）大不少，
    **W 与上下文预算的权衡留到第 9 步调**
- 冒烟（dev 测试集 5 题）：路由 5/5；Tradition 利润率题融合 #1+#2 恰好 =
  收入表(v=1.0) + 利润调节表(b=1.0)（两路各自的 best 被融合到顶部，
  算 margin 正好都要）；Holley M&A 题 #1 双路满分；Mercia M&A 题召回平淡，
  该题疑似不可答题（考第 8 步不编造）
- 复跑：`.venv\Scripts\python.exe -m enterprise_rag.retrieval.retriever "问题"`
  （--k 30 --vw 0.6 --top-n 20 --parent pages|els_window --window 15 --show 10）

## 索引阶段结果（第 5 步完成，2026-09-07）

- dev 集 10 份 → `data/indexes/`（不进 git）：每文档一对 `{doc}.faiss` +
  `{doc}.pkl`，外加 `index_meta.json`（模型/维度/时间/每文档 chunk 数，供
  A/B 目录区分与陈旧检测）。dev 全量 **1.24M token、18 秒**建完（免费）
- 嵌入模型：**硅基流动 BAAI/bge-m3**（免费档 2,000 RPM / 500K TPM；固定 1024 维，
  **不支持 dimensions 参数**——传了报 400 code 20015）。`.env` 配
  `EMBED_API_KEY / EMBED_BASE_URL / EMBED_MODEL`；百炼 `DASHSCOPE_API_KEY`
  留作 text-embedding-v4 A/B 备选（质量更强，¥0.5/M，实测稳定）
- 向量索引：**FAISS IndexFlatIP**（归一化后内积=余弦，**精确 KNN、召回 100%**），
  内部序号 == chunk id。刻意不用 ANN/向量数据库：语料静态、单机离线、按公司
  路由已由"每文档一索引"实现，实验迭代=换 `--out` 目录重跑、A/B 并存（决策
  讨论 2026-09-07）
- BM25：rank_bm25 BM25Okapi，语料序同 chunk id；分词保留千分位/小数
  （"86.6"、"1,234" 整 token，利于数字型查询）
- 批量 **token 预算制**（32 条 × 3,200 tok/请求双约束）：起因是百炼
  qwen3.7-text-embedding-flash 有隐藏单请求 ~4.5K token 上限且**超限不报错
  直接挂死**（排障半天才定位，已弃用该模型）；预算切批对任何隐藏上限都有防御
- `Embedder` 可复用类（建库与第 6 步查询端共用，保证同模型同维度）；客户端
  `max_retries=0`（避免与自定义重试叠加拖到十几分钟）+ `trust_env=False`
  （国内端点直连，绕过 Windows 系统代理）
- 复跑：`.venv\Scripts\python.exe -m enterprise_rag.indexing.ingestor`
  （按文档断点续跑；--force 重建；--limit 冒烟；eval 100 份同命令换输入目录，
  TPM 500K 下预计 ~30 分钟）
- 抽查（dev 测试集 5 题）：完整性 10/10（ntotal/维度/范数/双路对齐）；双路 top3
  重叠仅 0-1/3 = 互补性强；向量路补同义词（"buyback"→"Share repurchase" 直命中）、
  BM25 补精确词，融合价值得到验证

## 切分阶段结果（第 4 步完成，2026-09-07）

- dev 集 10 份 → `data/parsed/chunked/*.json`：`{metainfo, pages[], chunks[]}`，共
  **4,694 chunks**（content 3,940 / serialized_table 873 / serialized_image 41）+ 1,145 页父文本
- 切分策略（原版用 langchain RecursiveCharacterTextSplitter 300/50，我们自实现并强化）：
  - **标题为硬边界**：一个 content 块不跨章节
  - 段落贪心合并到 ≤300 token（小段落单独成块上下文太稀，合并自带语境）；
    只有单段超长才降级按行/词硬切，断口保留 50 token overlap——"结构优先、大小封顶"
  - 每个 content 块带**标题面包屑前缀**（如 "CHAIRMAN'S STATEMENT > Pension Scheme"），
    提高块的自含性（表格序列化思想的正文版，零 LLM 成本）
  - 表/vl 图不切碎：一张序列化表 = 一整块（同原版），一张数据图转写 = 一整块
- chunk 元数据带 `els: [start, end)`（原内容流元素区间）：PDF 分页是排版单位不是
  语义单位，第 6 步可用**跨页元素窗口**（而非页）构建父上下文；`pages[]` 同时保留
  （页级父文档 + 引用锚点）
- 质量：content 块 tokens p50=162/p95=288/max=335（超 300 仅 3/3,940，拼接开销）；
  els 单调、无空块；表块最大 5,739 tok（GMREIT 超大表，整块是设计使然）
- token 计数：tiktoken o200k_base（与原版一致；对 DeepSeek 是近似，仅用于切块）
- 复跑：`.venv\Scripts\python.exe -m enterprise_rag.processing.text_splitter`
  （默认 serialized → chunked；--chunk-size/--overlap 可调）

## 图片序列化阶段结果（第 3.5 步完成，2026-09-07）

- dev 集 **197/197 图片全部处理、0 失败**；41 张（21%）携带数据（图表/表图/信息图）
  完成文字转写，其余 156 张（照片/logo）只留分类标记
- 产物：`data/parsed/serialized/*.json` 的 picture 元素挂 `vl` 字段：
  `{type, data(HAS_DATA/NO_DATA), text(转写文本，NO_DATA 为空), img(裁剪图相对路径)}`
- 审计先行（同日）：全量轻量分类摸清分布；img_path 与内容流 picture 元素按
  **顺序位置**对齐（content_list 的 image 条目 ↔ content 流，10/10 文档验证）
- 做法：两阶段——①轻量分类（严格两行格式，**显式关 thinking**）；②仅 HAS_DATA
  做完整转写：只转写图上**印出的数字**（禁止按网格线估值，实测模型会"估"出
  似是而非的值且会遵守此禁令），公司名 + caption + 同页前后文注入
- 模型：`deepseek-v4-flash-vision-exp`（.env 的 `LLM_VISION_MODEL`，key/base_url 复用）。
  坑：这是混合推理模型，thinking 开着且 max_tokens 小时会返回**空 content**（分类阶段
  因此必须关 thinking）
- 成本：全量 ~90K 入/~51K 出 tokens（**<¥1**），8 并发约 2 分钟；eval 100 份外推约 ¥5
- 复跑/续跑：`.venv\Scripts\python.exe -m enterprise_rag.processing.image_serializer`
  （已带 vl 字段的图自动跳过；--limit 冒烟）

## 序列化阶段结果（第 3 步完成，2026-09-07）

- dev 集 873/873 表全部序列化，共 **6,262 个上下文独立信息块**（平均 7.2 块/表）
- 产物：`data/parsed/serialized/*.json`（在 docs 基础上：文本规范化 + 每表挂 `serialized`
  字段：`{subject_core_entities_list, relevant_headers_list, information_blocks[]}`）
- 做法：同页前后正文 + 表格 HTML + 公司名注入 → DeepSeek（temp=0，json_object 模式
  + pydantic 校验 + 3 次重试），忠实参考 IlyaRice tables_serialization.py
- 规范化顺带修复：fi/fl 连字断痕（"fi scal"→"fiscal"）、控制字符、零宽字符
- 超大表（>50 行或 html>12KB）追加"合并行压缩块数"指令，避免 max_tokens=8192 截断
  （dev 集仅 2 张此类表，Global_Medical_REIT t45/t50）
- 断点续跑按**表粒度**；总成本约 ¥5（~1.1M 入 + ~1.0M 出 tokens），全程 ~8 分钟
- 复跑/续跑：`.venv\Scripts\python.exe -m enterprise_rag.processing.table_serializer`
  （默认 data/parsed/docs → data/parsed/serialized；--limit 冒烟）
- LLM 配置在 `.env`（DEEPSEEK_API_KEY / LLM_BASE_URL / LLM_MODEL，不进 git）

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
- **公司名别名修正（2026-09-07，多模态审计时发现）**：两份 dev 文档曾按文件名误标——
  `Sandwell_Aquatics_Centre.pdf` 实为 **Billington Holdings plc** FY2022 年报（"水上运动中心"
  是其承建的钢结构项目，报告里只是项目照片）；`TSX_Y.pdf` 实为 **Yellow Pages Limited**
  （文件名来自 TSX 代码 Y）。csv 与 docs/serialized 的 company 已改，文件名/doc_id 保持不动
  （仅作 ID）；两份文档 149 张表已用正确公司名重序列化（旧名在信息块中出现 0 次）。
  官方 100 题不涉及这两家；dev 陷阱题（问 "Sandwell Aquatics Centre" 的 R&D/COO）正确答案仍为 N/A
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
