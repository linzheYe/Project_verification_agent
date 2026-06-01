# Project Verification Agent

This repository contains a prefetch pipeline for topic evidence collection and grounding.

## 1) Prefetch Quick Start

Main entry script:
- `runs/run_prefetch_topic_evidence.py`

### Step 1: Install dependencies

```bash
pip install requests ddgs trafilatura json-repair
```

### Step 2: Prepare input JSONL

Each line is one topic object. Required/used fields are controlled in `runs/run_prefetch_topic_evidence.py`:
- Query field: `QUERY_FIELD` (default `problem`), fallback `QUERY_FIELD_FALLBACKS` (default `query`)
- Topic id field: `TOPIC_ID_FIELD` (default `original_index`)
- Optional URL field: `URL_FIELD` (default `urls`, used when `USE_INPUT_URLS=True`)

Example line:

```json
{"original_index":"1","problem":"Who discovered penicillin?","urls":["https://www.britannica.com/biography/Alexander-Fleming"]}
```

### Step 3: Configure run parameters

Edit `runs/run_prefetch_topic_evidence.py`:
- `INPUT_JSONL`: input path
- `OUTPUT_DIR`: output directory
- `LLM_MODEL`: model name
- `USE_INPUT_URLS`: whether to use input URLs as trusted external sources
- `ENABLE_TOPIC_PARALLEL`, `TOPIC_PARALLEL_WORKERS`: parallel settings
- `OUTPUT_CONFLICT_POLICY`: `fail | backup | overwrite`

Retrieval scale:
- `WIKI_QUERY_COUNT`
- `WEB_QUERY_COUNT`
- `SNIPPETS_PER_QUERY`

### Step 4: Run

```bash
python runs/run_prefetch_topic_evidence.py
```

### Step 5: Check outputs

Under `OUTPUT_DIR`, key files are:
- `topic_grounding.jsonl`: structured evidence + topic brief
- `topic_guidance.jsonl`: topic-level guidance summary
- `fetched_fulltext_pages.jsonl`: fetched full-text pages merged into final pool
- `topic_grounding_trace.jsonl`: full intermediate trace for reproducibility/debug
- `run_prefetch_topic_evidence_report.json`: run-level statistics

## 2) Prefetch Flow (What It Does)

Pipeline stages:
1. Query planning and DDG retrieval
2. External URL full-text fetch (if `USE_INPUT_URLS=True`)
3. Candidate cleaning/deduplication
4. LLM snippet-level URL selection (search candidates)
5. Full-text fetch for selected URLs
6. Evidence extraction on merged full-text pool

Two run modes:
- With external URLs: trusted input URLs are fetched directly and merged with selected search evidence
- Without external URLs: pure search flow (retrieve -> select -> fetch -> extract)

Detailed flow reference:
- `mds/prefetch_flow_with_and_without_external_urls.md`

## 3) External Package Dependencies

Detected third-party packages in this repository:

1. `requests`
- Used for HTTP calls in LLM/web-related steps.
- Seen in: `scripts/llm_api.py`, `scripts/prefetch_topic_evidence_stages.py`

2. `ddgs`
- DuckDuckGo retrieval backend (`DDGS`) for snippet search.
- Seen in: `scripts/ddg_search.py`

3. `trafilatura`
- Full-text extraction/cleaning from fetched web pages.
- Seen in: `scripts/prefetch_topic_evidence_stages.py`

4. `json_repair`
- Tolerant JSON parsing for imperfect model outputs.
- Seen in: `scripts/data_io.py`

Dependency reference doc:
- `mds/README_external_packages.md`
