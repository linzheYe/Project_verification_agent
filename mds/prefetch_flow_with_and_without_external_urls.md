# Prefetch 流程说明（含外加 URL / 不含外加 URL）

## 摘要式流程概括（面向论文描述）
本流程的目标是为每个问题（topic）构建一个高质量、可追溯的证据池，并在此基础上生成结构化证据与主题摘要。整体设计遵循“可信外部证据优先 + 检索证据补充 + 全文级证据抽取”的原则。

具体来说，流程首先围绕原始问题自动生成多个检索表达，以提高信息召回的覆盖面；随后通过 Web 检索获取候选片段，并进行两层去重（URL 层和语义近重复层），减少冗余与噪声。在存在外加 URL 的情况下，外加 URL 被视为先验可信来源，不与检索候选竞争片段级筛选，而是直接抓取全文并进入候选证据池。这样可以保证关键先验证据不会被筛选步骤误伤。

在检索分支中，系统将去重后的片段候选提交给 LLM 做相关性筛选，保留最可能支持问题回答的 URL，再抓取其网页全文。最终，系统将“外加 URL 全文”和“检索筛选 URL 全文”合并为统一证据池，并执行 URL 级去重（外加来源优先保留）。在统一证据池上，LLM 输出细粒度证据项（引用段落、证据摘要）和 topic-level 的简要结论。

该设计的核心优点有三点：
1. 召回与精度兼顾：检索扩展保证广覆盖，LLM 筛选抑制噪声。
2. 证据可解释：最终输出绑定 URL、证据文本与摘要，支持审计和复核。
3. 工程稳健：支持并行处理、抓取重试、增量可观测输出，适合批量实验与论文复现实验。

---

## 1. 输入与输出（概念层）

### 输入
每行 JSONL 对应一个 topic，至少包含：
- 问题字段（例如 `original_question`）
- 可选外加 URL 字段（例如 `simpleqa_urls`）
- 建议包含稳定 id 字段（例如 `original_index`）

### 输出
每个 topic 会产生四类结果：
- `topic_grounding.jsonl`：证据项与 topic 摘要
- `topic_guidance.jsonl`：topic 级 guidance 摘要
- `fetched_fulltext_pages.jsonl`：最终并入证据池的抓取全文页
- `topic_grounding_trace.jsonl`：完整中间过程轨迹（便于分析与复现）

---

## 2. 五阶段主流程及每一步目的

### Stage 1: Query Planning + Web Retrieval
**目的**：把单个问题扩展为多个检索视角，提高召回率。  
做法：LLM 生成 wiki/web 检索词；对每个检索词调用 DDG，得到原始 snippet 候选。

### Stage 2: External URL Fulltext Fetch (Trusted Source)
**目的**：保留先验可信证据，不让其在片段筛选中被误删。  
做法：当提供外加 URL 时，直接抓取其全文（trafilatura），并记录抓取状态（成功/失败）。

### Candidate Preparation (Shared Pre-selection Processing)
**目的**：降低冗余和噪声，提高后续 LLM URL 筛选效率。  
做法：
1. URL 去重（同 URL 仅保留一条）
2. 语义近重复去重（基于 title+snippet 的相似度）
3. 删除与外加 URL 重复的检索候选（避免重复抓取与重复证据）

### Stage 3: LLM Snippet-level URL Selection (Search-only)
**目的**：从检索候选中挑出最有支持价值的 URL。  
做法：只对“检索候选”做 LLM 选择（外加 URL 不参与此阶段）。

### Stage 4: Fulltext Fetch for Selected Search URLs
**目的**：将片段级候选升级为全文级证据。  
做法：对 Stage 3 选中的 URL 抓取全文；失败可重试（有限次数）。

### Stage 5: Evidence Extraction on Merged Fulltext Pool
**目的**：在统一证据池中生成结构化证据和 topic 摘要。  
做法：合并 external 与 search 两路全文，按 URL 去重（external-first），再由 LLM 输出 `evidence_items` 与 `topic_brief`。

---

## 3. 使用外加 URL 与不使用外加 URL 的差异

### A. 使用外加 URL（`USE_INPUT_URLS=True`）
- 读取 `URL_FIELD`（如 `simpleqa_urls`）
- Stage 2 直接抓取外加 URL 全文
- 检索候选会排除与外加 URL 重复者
- 最终证据池 = external 全文 + selected search 全文（external-first 去重）

### B. 不使用外加 URL（`USE_INPUT_URLS=False`）
- `external_urls` 为空
- Stage 2 产出为空
- 流程退化为纯检索路径：检索 -> 筛选 -> 抓全文 -> 抽证据

---

## 4. 当前实现中的关键配置（与复现相关）

以下字段位于 `runs/run_prefetch_topic_evidence.py`：

### 输入与字段映射
- `INPUT_JSONL`：输入 JSONL 路径
- `QUERY_FIELD` / `QUERY_FIELD_FALLBACKS`：问题字段映射
- `URL_FIELD`：外加 URL 字段映射
- `TOPIC_ID_FIELD`：topic id 字段（建议 `original_index`）
- `REQUIRE_TOPIC_ID_FROM_INPUT = True`：强制每行必须提供 `TOPIC_ID_FIELD`，保证输出 `topic_id` 与上游体系可对齐

### 外加 URL 与检索规模
- `USE_INPUT_URLS`：是否启用外加 URL
- `WIKI_QUERY_COUNT` / `WEB_QUERY_COUNT`：检索词数量
- `SNIPPETS_PER_QUERY`：每个检索词的 DDG 返回条数

### 并行与可观测性
- `ENABLE_TOPIC_PARALLEL` / `TOPIC_PARALLEL_WORKERS`：topic 级并行
- `PRINT_PROGRESS`：运行过程打印开关

### 输出安全策略
- `OUTPUT_CONFLICT_POLICY`：`fail | backup | overwrite`
  - `fail`：发现同名输出文件即停止，防止覆盖
  - `backup`：自动备份旧文件再写新结果
  - `overwrite`：直接覆盖

---

## 5. 运行行为与工程特性

1. **流式落盘**：每完成一个 topic 就立刻 append 写入输出 JSONL，不再等全部完成后统一写入。  
2. **并行可追踪**：并行模式下会打印 `DONE k/N topic_id=...`，便于定位慢任务。  
3. **失败隔离**：单 topic 失败不会中断整批任务，最终报告包含 `failed_topic_count`。  
4. **抓取稳健性**：网页抓取支持有限重试，失败会保留 `fetch_error` 供后续分析。

---

## 6. 论文写作可引用的流程要点（简版）

可将本方法概括为：
- “先验可信 URL 直抓全文保底 + 检索候选 LLM 选择补充 + 全文级统一证据抽取”。
- 外加 URL 不参与片段竞争，而在全文层与检索结果融合。
- 通过分阶段去重、并行处理和流式输出，兼顾证据质量、运行效率与实验可复现性。
