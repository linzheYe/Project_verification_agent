from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is importable even when running this file by absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.prefetch_topic_evidence_pipeline import PrefetchPipelineConfig, PrefetchTopicEvidencePipeline
from scripts.data_io import read_jsonl
from scripts.prefetch_topic_evidence_stages import PrefetchStageConfig

# =========================
# User-configurable settings
# =========================
# All paths below are relative to project root: /home/an/Project_verification_agent

INPUT_JSONL = "prefetch_data/topics_select/simpleqa_select_4topics.jsonl"
OUTPUT_DIR = "prefetch_data/topics_select/try2_2"
LLM_MODEL = "openai/gpt-5.4-nano"
USE_INPUT_URLS = False
ENABLE_TOPIC_PARALLEL = True
TOPIC_PARALLEL_WORKERS = 10
PRINT_PROGRESS = True
OUTPUT_CONFLICT_POLICY = "fail"  # one of: fail, backup, overwrite
#  - fail: 发现输出目录里已有同名结果文件时，直接报错并停止，防止覆盖。
#  - backup: 先把已有同名文件重命名为 .bak_时间戳，再继续运行写新文件。
#  - overwrite: 不做保护，直接用新结果覆盖同名文件。

QUERY_FIELD = "problem" # primary field name to read query text
QUERY_FIELD_FALLBACKS = ["simpleqa_problem"] # fallback field names if primary field is missing/empty
TOPIC_ID_FIELD = "original_index"
URL_FIELD = "urls"
REQUIRE_TOPIC_ID_FROM_INPUT = True
TOPIC_START_INDEX = 3  # 0-based inclusive start index; the row at this index is included in the run.
LIMIT_TOPICS: int | None = 2
# Example: to run rows 4-6 only, use TOPIC_START_INDEX = 3 and LIMIT_TOPICS = 3.

# Retrieval scale configuration:
# - WIKI_QUERY_COUNT / WEB_QUERY_COUNT: number of generated search queries.
# - SNIPPETS_PER_QUERY: top-N DDG snippets fetched per query.
WIKI_QUERY_COUNT = 3
WEB_QUERY_COUNT = 3
SNIPPETS_PER_QUERY = 5

# Trace output quick reference (topic_grounding_trace.jsonl):
# - One row per topic.
# - Core fields:
#   - topic_id, query, claims
#   - search_queries: generated query terms (query_id/query_text)
#   - raw_search_candidates: DDG candidates before cleaning
#   - search_candidates_for_selection: deduped candidates used for LLM URL selection
#   - selected_search_urls: URLs picked by LLM snippet filtering
#   - external_fetched_pages / selected_search_fetched_pages / merged_fetched_pages:
#     lightweight fetch metadata (no full_text), including:
#       url_id, source_url, source_title, source_type (external|search),
#       fetch_error, text_char_len, is_truncated, max_page_text_chars
#   - evidence_items, topic_brief: final evidence + summary



# =========================
# =========================
def _project_root() -> Path:
    # runs/run_prefetch_topic_evidence.py -> project root is parent of "runs"
    return PROJECT_ROOT


def _read_topic_rows(input_jsonl: Path, use_input_urls: bool) -> list[dict[str, Any]]:
    """Read input JSONL where each line is one topic.

    Query field priority:
    1) QUERY_FIELD
    2) QUERY_FIELD_FALLBACKS (in order)

    Optional URL field:
    - URL_FIELD (used only when use_input_urls=True)
    """
    rows = read_jsonl(input_jsonl)
    topic_rows: list[dict[str, Any]] = []
    for row in rows:
        query = ""
        query_fields = [QUERY_FIELD] + [field for field in QUERY_FIELD_FALLBACKS if field != QUERY_FIELD]
        for field_name in query_fields:
            value = str(row.get(field_name) or "").strip()
            if value:
                query = value
                break
        if not query:
            continue

        external_urls: list[str] = []
        if use_input_urls:
            external_urls = [str(url).strip() for url in (row.get(URL_FIELD) or []) if str(url).strip()]

        topic_id = str(row.get(TOPIC_ID_FIELD) or "").strip()
        if REQUIRE_TOPIC_ID_FROM_INPUT and not topic_id:
            raise ValueError(
                f"Missing required topic id field '{TOPIC_ID_FIELD}' in one input row. "
                "Set REQUIRE_TOPIC_ID_FROM_INPUT=False if you want fallback IDs."
            )

        topic_rows.append(
            {
                "topic_id": topic_id,
                "query": query,
                "external_urls": external_urls,
            }
        )
    return topic_rows


def _progress(message: str) -> None:
    if PRINT_PROGRESS:
        print(f"[prefetch] {message}", flush=True)


def _prepare_output_dir_and_check_conflicts(output_dir: Path, conflict_policy: str) -> None:
    """Create output dir if needed and handle existing output file conflicts safely."""
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_output_files = [
        output_dir / "topic_grounding.jsonl",
        output_dir / "topic_guidance.jsonl",
        output_dir / "fetched_fulltext_pages.jsonl",
        output_dir / "topic_grounding_trace.jsonl",
        output_dir / "run_prefetch_topic_evidence_report.json",
    ]
    existing_files = [path for path in expected_output_files if path.exists()]
    if not existing_files:
        return

    policy = str(conflict_policy).strip().lower()
    if policy == "overwrite":
        _progress(f"OUTPUT_CONFLICT policy=overwrite existing_files={len(existing_files)}")
        return
    if policy == "backup":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for path in existing_files:
            backup_path = path.with_name(f"{path.name}.bak_{timestamp}")
            path.rename(backup_path)
        _progress(f"OUTPUT_CONFLICT policy=backup backed_up_files={len(existing_files)}")
        return
    if policy == "fail":
        conflict_list = ", ".join(str(path.name) for path in existing_files)
        raise FileExistsError(
            f"Output conflict in {output_dir}: existing files [{conflict_list}]. "
            "Change OUTPUT_DIR or set OUTPUT_CONFLICT_POLICY to backup/overwrite."
        )
    raise ValueError(f"Invalid OUTPUT_CONFLICT_POLICY: {conflict_policy}. Use fail/backup/overwrite.")


def main() -> None:
    root = _project_root()
    input_jsonl = root / INPUT_JSONL
    output_dir = root / OUTPUT_DIR
    _progress("BOOT run_prefetch_topic_evidence.py")
    _progress(f"PATHS input_jsonl={input_jsonl} output_dir={output_dir}")

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    _prepare_output_dir_and_check_conflicts(output_dir, OUTPUT_CONFLICT_POLICY)

    topic_rows = _read_topic_rows(input_jsonl, use_input_urls=bool(USE_INPUT_URLS))
    start_index = max(0, int(TOPIC_START_INDEX))
    topic_rows = topic_rows[start_index:]
    if LIMIT_TOPICS is not None:
        topic_rows = topic_rows[: max(0, int(LIMIT_TOPICS))]
    _progress(
        "INPUT_LOADED "
        f"topic_count={len(topic_rows)} "
        f"topic_start_index={start_index} "
        f"limit_topics={LIMIT_TOPICS} "
        f"use_input_urls={USE_INPUT_URLS}"
    )

    config = PrefetchPipelineConfig(
        llm_model=LLM_MODEL,
        enable_topic_parallel=bool(ENABLE_TOPIC_PARALLEL),
        topic_parallel_workers=int(TOPIC_PARALLEL_WORKERS),
        print_progress=bool(PRINT_PROGRESS),
        stage_config=PrefetchStageConfig(
            wiki_query_count=int(WIKI_QUERY_COUNT),
            web_query_count=int(WEB_QUERY_COUNT),
            snippets_per_query=int(SNIPPETS_PER_QUERY),
        ),
    )

    pipeline = PrefetchTopicEvidencePipeline(config=config, output_dir=output_dir)
    report = pipeline.run(topic_rows=topic_rows)

    report_path = output_dir / "run_prefetch_topic_evidence_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(f"REPORT_WRITTEN path={report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
