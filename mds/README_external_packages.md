# Project External Packages

This document summarizes third-party Python packages used in `/home/an/Project_verification_agent` based on source-code import scanning.

## Summary

Detected external packages:

1. `requests`
2. `ddgs`
3. `trafilatura`
4. `json_repair`

## Package Details

### 1) `requests`
- Purpose in project:
  - HTTP requests to external APIs/pages.
- Import locations:
  - `scripts/llm_api.py`
  - `scripts/prefetch_topic_evidence_stages.py`
  - `temp/pipeline_prefetch_topic_evidence.py`

### 2) `ddgs`
- Purpose in project:
  - DuckDuckGo search retrieval (`DDGS`) for evidence collection.
- Import locations:
  - `scripts/ddg_search.py`
- Notes:
  - The code explicitly checks this dependency and prompts installation when missing.

### 3) `trafilatura`
- Purpose in project:
  - Web content extraction / cleaning from fetched HTML pages.
- Import locations:
  - `scripts/prefetch_topic_evidence_stages.py`
  - `temp/pipeline_prefetch_topic_evidence.py`
- Notes:
  - Imported lazily in workflow steps; failures may trigger fallback behavior.

### 4) `json_repair`
- Purpose in project:
  - Repair and parse imperfect JSON outputs.
- Import locations:
  - `scripts/data_io.py`
- Notes:
  - Imported lazily; used as a tolerant parse mode.

## Suggested Installation

```bash
pip install requests ddgs trafilatura json-repair
```


