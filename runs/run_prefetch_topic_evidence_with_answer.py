from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure project root is importable even when running this file by absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.prefetch_topic_evidence_pipeline import PrefetchPipelineConfig
from scripts.data_io import parse_json_object_robust, read_jsonl, write_jsonl
from scripts.llm_api import LLMClient, LLMConnectivityError
from scripts.prefetch_topic_evidence_stages import (
    PrefetchStageConfig,
    as_jsonable_rows,
    merge_fetched_pages_external_first,
    prepare_search_candidates_for_selection,
    stage1_plan_and_search,
    stage2_prepare_external_fulltext,
    stage3_select_search_urls,
    stage4_fetch_selected_search_fulltext,
)
from scripts.prompt_template import build_topic_grounding_evidence_extract_with_answer_prompt

# =========================
# User-configurable settings
# =========================
# All paths below are relative to project root: /home/an/Project_verification_agent

INPUT_JSONL =  "response_data/question_answer/accepted.whitelist_plus_tmp_filter.id_question_answer_url_only.jsonl"
OUTPUT_DIR = "prefetch_data/yue_question/try15_changed_4to13"
LLM_MODEL = "openai/gpt-5.4-nano"
USE_INPUT_URLS = True
REUSE_FETCHED_FULLTEXT_JSONL = "prefetch_data/yue_question/try7_4to13/fetched_fulltext_pages.jsonl"  
# e.g. "prefetch_data/yue_question/try1/fetched_fulltext_pages.jsonl"
ENABLE_TOPIC_PARALLEL = True
TOPIC_PARALLEL_WORKERS = 10
PRINT_PROGRESS = True
OUTPUT_CONFLICT_POLICY = "fail"  # one of: fail, backup, overwrite

QUERY_FIELD = "question"  # primary field name to read query text
QUERY_FIELD_FALLBACKS = []
ANSWER_FIELD = "answer"
ANSWER_FIELD_FALLBACKS = []
TOPIC_ID_FIELD = "id"
URL_FIELD = "url"
# False means rows without `id` will be assigned fallback topic IDs like T1/T2.
# Keep this False only for ad-hoc inputs where stable source IDs are not required;
# set it to True when output must preserve input IDs for reuse, trace comparison, or joins.
REQUIRE_TOPIC_ID_FROM_INPUT = True
REQUIRE_ANSWER_FROM_INPUT = True
TOPIC_START_INDEX = 3  # 0-based inclusive start index; the row at this index is included in the run.
LIMIT_TOPICS: int | None = 10
# Example: to run rows 4-6 only, use TOPIC_START_INDEX = 3 and LIMIT_TOPICS = 3.

WIKI_QUERY_COUNT = 3
WEB_QUERY_COUNT = 3
SNIPPETS_PER_QUERY = 5
STAGE5_MAX_ATTEMPTS = 3


def _project_root() -> Path:
    return PROJECT_ROOT


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_snippet_ids_from_text(text: str) -> list[str]:
    found = re.findall(r"\bS\d+\b", str(text or "").upper())
    ordered: list[str] = []
    seen: set[str] = set()
    for snippet_id in found:
        if snippet_id in seen:
            continue
        seen.add(snippet_id)
        ordered.append(snippet_id)
    return ordered


def _validate_snippet_references(
    *,
    topic_brief: str,
    answer_reasoning: str,
    valid_snippet_ids: set[str],
) -> None:
    topic_brief_refs = _extract_snippet_ids_from_text(topic_brief)
    reasoning_refs = _extract_snippet_ids_from_text(answer_reasoning)

    missing_topic_brief_refs = [snippet_id for snippet_id in topic_brief_refs if snippet_id not in valid_snippet_ids]
    missing_reasoning_refs = [snippet_id for snippet_id in reasoning_refs if snippet_id not in valid_snippet_ids]

    if not missing_topic_brief_refs and not missing_reasoning_refs:
        return

    parts: list[str] = []
    if missing_topic_brief_refs:
        parts.append(
            "topic_brief references missing snippet_ids: " + ", ".join(missing_topic_brief_refs)
        )
    if missing_reasoning_refs:
        parts.append(
            "answer_assessment.reasoning references missing snippet_ids: "
            + ", ".join(missing_reasoning_refs)
        )
    raise RuntimeError("; ".join(parts))


def _build_guidance_evidence_items(evidence_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guidance_items: list[dict[str, Any]] = []
    for item in evidence_items:
        guidance_items.append(
            {
                "snippet_id": str(item.get("snippet_id") or "").strip(),
                "source_url": str(item.get("source_url") or "").strip(),
                "source_title": str(item.get("source_title") or "").strip(),
                "content": str(item.get("content") or "").strip(),
            }
        )
    return guidance_items


def _call_llm_json(llm: LLMClient, system_prompt: str, user_prompt: str, *, stage_name: str) -> dict[str, Any]:
    try:
        raw = llm.call(system_prompt, user_prompt, temperature=0.2)
    except LLMConnectivityError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{stage_name}_llm_call_failed: {exc}") from exc

    try:
        parsed, _ = parse_json_object_robust(raw)
        return parsed
    except Exception as exc:
        raise RuntimeError(f"{stage_name}_llm_parse_failed: {exc}") from exc


def _validate_stage5_payload(llm_json: dict[str, Any]) -> None:
    evidence_items = llm_json.get("evidence_items")
    topic_brief = llm_json.get("topic_brief")
    answer_assessment = llm_json.get("answer_assessment")
    if not isinstance(evidence_items, list):
        raise RuntimeError("stage5_missing_or_invalid_evidence_items")
    if not isinstance(topic_brief, str):
        raise RuntimeError("stage5_missing_or_invalid_topic_brief")
    if not isinstance(answer_assessment, dict):
        raise RuntimeError("stage5_missing_or_invalid_answer_assessment")


def _read_topic_rows(input_jsonl: Path, use_input_urls: bool) -> list[dict[str, Any]]:
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

        answer = ""
        answer_fields = [ANSWER_FIELD] + [field for field in ANSWER_FIELD_FALLBACKS if field != ANSWER_FIELD]
        for field_name in answer_fields:
            value = str(row.get(field_name) or "").strip()
            if value:
                answer = value
                break
        if REQUIRE_ANSWER_FROM_INPUT and not answer:
            raise ValueError(
                f"Missing required answer field '{ANSWER_FIELD}' in one input row. "
                "Set REQUIRE_ANSWER_FROM_INPUT=False if you want to allow empty answers."
            )

        external_urls: list[str] = []
        if use_input_urls:
            raw_urls = row.get(URL_FIELD)
            if isinstance(raw_urls, list):
                external_urls = [str(url).strip() for url in raw_urls if str(url).strip()]
            else:
                single_url = str(raw_urls or "").strip()
                external_urls = [single_url] if single_url else []

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
                "answer": answer,
                "external_urls": external_urls,
            }
        )
    return topic_rows


def _progress(message: str) -> None:
    if PRINT_PROGRESS:
        print(f"[prefetch-answer] {message}", flush=True)


def _prepare_output_dir_and_check_conflicts_for_paths(
    output_dir: Path, conflict_policy: str, expected_output_files: list[Path]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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


def _prepare_output_dir_and_check_conflicts(output_dir: Path, conflict_policy: str) -> None:
    _prepare_output_dir_and_check_conflicts_for_paths(
        output_dir,
        conflict_policy,
        [
            output_dir / "topic_guidance_with_answer.jsonl",
            output_dir / "fetched_fulltext_pages.jsonl",
            output_dir / "topic_grounding_trace_with_answer.jsonl",
            output_dir / "run_prefetch_topic_evidence_with_answer_report.json",
        ],
    )


def _extract_evidence_with_answer_assessment(
    *,
    llm: LLMClient,
    topic_id: str,
    query: str,
    answer: str,
    merged_fetched_pages: list[Any],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    usable_pages = [page for page in merged_fetched_pages if not page.fetch_error and page.full_text.strip()]
    if not usable_pages:
        return [], "", {}

    pages_for_prompt = [
        {
            "url_id": page.url_id,
            "source_url": page.source_url,
            "source_title": page.source_title,
            "candidate_ids": list(page.candidate_ids),
            "query_ids": list(page.query_ids),
            "full_text": page.full_text,
            "fetch_error": page.fetch_error,
            "source_type": page.source_type,
        }
        for page in usable_pages
    ]

    system_prompt, user_prompt = build_topic_grounding_evidence_extract_with_answer_prompt(
        {
            "query": query,
            "answer": answer,
            "pages": pages_for_prompt,
        }
    )
    llm_json: dict[str, Any] | None = None
    attempt_errors: list[str] = []
    for attempt_index in range(1, max(1, int(STAGE5_MAX_ATTEMPTS)) + 1):
        attempt_system_prompt = system_prompt
        if attempt_index > 1:
            attempt_system_prompt = (
                "Previous output was invalid or incomplete. Return only one valid JSON object in the required schema.\n\n"
                + system_prompt
            )
        try:
            candidate_json = _call_llm_json(
                llm,
                attempt_system_prompt,
                user_prompt,
                stage_name="stage5_extract_evidence_with_answer",
            )
            _validate_stage5_payload(candidate_json)
            llm_json = candidate_json
            break
        except RuntimeError as exc:
            attempt_errors.append(f"attempt_{attempt_index}:{exc}")
    if llm_json is None:
        raise RuntimeError("; ".join(attempt_errors))

    valid_url_ids = {page.url_id for page in usable_pages}
    evidence_items: list[dict[str, Any]] = []
    evidence_index = 0

    for raw_item in (llm_json.get("evidence_items") or []):
        if not isinstance(raw_item, dict):
            continue

        url_id = _normalize_whitespace(raw_item.get("url_id") or "")
        relevant_text = str(raw_item.get("relevant_text") or "").strip()

        if not url_id or url_id not in valid_url_ids or not relevant_text:
            continue

        source_page = next((page for page in usable_pages if page.url_id == url_id), None)
        if source_page is None:
            continue

        evidence_index += 1
        evidence_items.append(
            {
                "snippet_id": f"S{evidence_index}",
                "topic_id": topic_id,
                "url_id": url_id,
                "source_url": source_page.source_url,
                "source_title": source_page.source_title,
                "source_type": source_page.source_type,
                "content": relevant_text,
            }
        )

    topic_brief = _normalize_whitespace(llm_json.get("topic_brief") or "")

    answer_assessment_raw = llm_json.get("answer_assessment") or {}
    if not isinstance(answer_assessment_raw, dict):
        answer_assessment_raw = {}
    answer_assessment = {
        "provided_answer": str(answer_assessment_raw.get("provided_answer") or answer).strip(),
        "reasoning": str(answer_assessment_raw.get("reasoning") or "").strip(),
        "is_supported": bool(answer_assessment_raw.get("is_supported")),
        "is_unique_answer": bool(answer_assessment_raw.get("is_unique_answer")),
        "canonical_answer": str(answer_assessment_raw.get("canonical_answer") or "").strip(),
    }

    valid_snippet_ids = {item["snippet_id"] for item in evidence_items}
    _validate_snippet_references(
        topic_brief=topic_brief,
        answer_reasoning=answer_assessment["reasoning"],
        valid_snippet_ids=valid_snippet_ids,
    )
    return evidence_items, topic_brief, answer_assessment


def _load_reused_fetched_pages(path: Path) -> dict[str, list[Any]]:
    rows = read_jsonl(path)
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        topic_id = str(row.get("topic_id") or "").strip()
        if not topic_id:
            continue
        grouped.setdefault(topic_id, []).append(
            SimpleNamespace(
                topic_id=topic_id,
                query=str(row.get("query") or "").strip(),
                url_id=str(row.get("url_id") or "").strip(),
                source_url=str(row.get("source_url") or "").strip(),
                source_title=str(row.get("source_title") or "").strip(),
                source_type=str(row.get("source_type") or "").strip(),
                candidate_ids=list(row.get("candidate_ids") or []),
                query_ids=list(row.get("query_ids") or []),
                fetch_error=row.get("fetch_error"),
                full_text=str(row.get("full_text") or ""),
            )
        )
    return grouped


def _page_to_jsonable_row(page: Any) -> dict[str, Any]:
    return {
        "topic_id": str(getattr(page, "topic_id", "") or ""),
        "query": str(getattr(page, "query", "") or ""),
        "url_id": str(getattr(page, "url_id", "") or ""),
        "source_url": str(getattr(page, "source_url", "") or ""),
        "source_title": str(getattr(page, "source_title", "") or ""),
        "source_type": str(getattr(page, "source_type", "") or ""),
        "candidate_ids": list(getattr(page, "candidate_ids", []) or []),
        "query_ids": list(getattr(page, "query_ids", []) or []),
        "fetch_error": getattr(page, "fetch_error", None),
        "full_text": str(getattr(page, "full_text", "") or ""),
    }


class PrefetchTopicEvidenceWithAnswerPipeline:
    def __init__(
        self,
        *,
        config: PrefetchPipelineConfig,
        output_dir: Path,
        reused_fetched_pages_by_topic: dict[str, list[Any]] | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.reused_fetched_pages_by_topic = reused_fetched_pages_by_topic or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, topic_rows: list[dict[str, Any]]) -> dict[str, Any]:
        topic_inputs = self._build_topic_inputs(topic_rows)
        total_topics = len(topic_inputs)
        self._progress(f"START total_topics={total_topics} parallel={self.config.enable_topic_parallel}")

        guidance_path = self.output_dir / "topic_guidance_with_answer.jsonl"
        fetched_path = self.output_dir / "fetched_fulltext_pages.jsonl"
        trace_path = self.output_dir / "topic_grounding_trace_with_answer.jsonl"
        self._initialize_output_files([guidance_path, fetched_path, trace_path])

        failed_topic_count = 0
        if self.config.enable_topic_parallel and len(topic_inputs) > 1:
            self._progress(
                f"SUBMIT mode=parallel workers={max(1, int(self.config.topic_parallel_workers))} total={total_topics}"
            )
            with ThreadPoolExecutor(max_workers=max(1, int(self.config.topic_parallel_workers))) as executor:
                future_to_topic_id = {
                    executor.submit(self._process_single_topic, topic): str(topic.get("topic_id") or "")
                    for topic in topic_inputs
                }
                completed = 0
                for future in as_completed(future_to_topic_id):
                    completed += 1
                    topic_id = future_to_topic_id[future]
                    try:
                        row = future.result()
                        self._append_completed_topic_row(
                            guidance_path=guidance_path,
                            fetched_path=fetched_path,
                            trace_path=trace_path,
                            row=row,
                        )
                        self._progress(f"DONE {completed}/{total_topics} topic_id={topic_id}")
                    except Exception as exc:
                        failed_topic_count += 1
                        self._progress(f"FAIL {completed}/{total_topics} topic_id={topic_id} error={type(exc).__name__}:{exc}")
                        row = self._build_failed_topic_row(topic_id=topic_id, error_text=str(exc))
                        self._append_completed_topic_row(
                            guidance_path=guidance_path,
                            fetched_path=fetched_path,
                            trace_path=trace_path,
                            row=row,
                        )
        else:
            self._progress(f"SUBMIT mode=serial total={total_topics}")
            completed = 0
            for topic in topic_inputs:
                completed += 1
                topic_id = str(topic.get("topic_id") or "")
                try:
                    row = self._process_single_topic(topic)
                    self._append_completed_topic_row(
                        guidance_path=guidance_path,
                        fetched_path=fetched_path,
                        trace_path=trace_path,
                        row=row,
                    )
                    self._progress(f"DONE {completed}/{total_topics} topic_id={topic_id}")
                except Exception as exc:
                    failed_topic_count += 1
                    self._progress(f"FAIL {completed}/{total_topics} topic_id={topic_id} error={type(exc).__name__}:{exc}")
                    row = self._build_failed_topic_row(topic_id=topic_id, error_text=str(exc))
                    self._append_completed_topic_row(
                        guidance_path=guidance_path,
                        fetched_path=fetched_path,
                        trace_path=trace_path,
                        row=row,
                    )
        self._progress(f"ALL_DONE success={len(topic_inputs) - failed_topic_count} failed={failed_topic_count}")

        return {
            "topic_count": len(topic_inputs),
            "failed_topic_count": failed_topic_count,
            "external_urls_enabled": any(bool(topic.get("external_urls")) for topic in topic_inputs),
            "topic_grounding_path": str(guidance_path),
            "topic_guidance_path": str(guidance_path),
            "fetched_fulltext_path": str(fetched_path),
            "topic_grounding_trace_path": str(trace_path),
        }

    def _build_topic_inputs(self, topic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        topic_inputs: list[dict[str, Any]] = []
        clean_rows = [row for row in topic_rows if str(row.get("query") or "").strip()]
        for topic_index, row in enumerate(clean_rows, start=1):
            query = str(row.get("query") or "").strip()
            topic_id = str(row.get("topic_id") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not topic_id:
                topic_id = f"T{topic_index}"
            topic_inputs.append(
                {
                    "topic_id": topic_id,
                    "query": query,
                    "answer": answer,
                    "claims": [{"claim_id": f"{topic_id}_C1", "claim_text": query}],
                    "external_urls": [str(url).strip() for url in (row.get("external_urls") or []) if str(url).strip()],
                }
            )
        return topic_inputs

    def _process_single_topic(self, topic: dict[str, Any]) -> dict[str, Any]:
        started_at = time.time()
        topic_id = topic["topic_id"]
        query = topic["query"]
        answer = topic["answer"]
        claims = topic["claims"]
        external_urls = topic["external_urls"]
        self._progress(f"TOPIC_START topic_id={topic_id}")
        llm = LLMClient(provider="openrouter", model=self.config.llm_model, timeout_seconds=60, max_retries=1)

        reused_fetched_pages = self.reused_fetched_pages_by_topic.get(topic_id)
        if reused_fetched_pages is not None:
            search_queries = []
            raw_search_candidates = []
            search_candidates_for_selection = []
            selected_search_urls = []
            external_fetched_pages = [page for page in reused_fetched_pages if str(getattr(page, "source_type", "")) == "external"]
            selected_search_fetched_pages = [
                page for page in reused_fetched_pages if str(getattr(page, "source_type", "")) != "external"
            ]
            merged_fetched_pages = list(reused_fetched_pages)
            self._progress(f"TOPIC_REUSE_FETCHED topic_id={topic_id} reused_pages={len(merged_fetched_pages)}")
        else:
            search_queries, raw_search_candidates = stage1_plan_and_search(
                llm=llm,
                topic_id=topic_id,
                query=query,
                claims=claims,
                config=self.config.stage_config,
            )
            search_candidates_for_selection = prepare_search_candidates_for_selection(
                raw_search_candidates=raw_search_candidates,
                external_urls=external_urls,
                config=self.config.stage_config,
            )
            external_fetched_pages = stage2_prepare_external_fulltext(
                topic_id=topic_id,
                query=query,
                external_urls=external_urls,
                config=self.config.stage_config,
            )
            selected_search_urls = stage3_select_search_urls(
                llm=llm,
                query=query,
                claims=claims,
                prepared_search_candidates=search_candidates_for_selection,
                config=self.config.stage_config,
            )
            selected_search_fetched_pages = stage4_fetch_selected_search_fulltext(
                topic_id=topic_id,
                query=query,
                selected_search_urls=selected_search_urls,
                config=self.config.stage_config,
            )
            merged_fetched_pages = merge_fetched_pages_external_first(
                external_fetched_pages=external_fetched_pages,
                selected_search_fetched_pages=selected_search_fetched_pages,
            )
        evidence_items, topic_brief, answer_assessment = _extract_evidence_with_answer_assessment(
            llm=llm,
            topic_id=topic_id,
            query=query,
            answer=answer,
            merged_fetched_pages=merged_fetched_pages,
        )
        guidance_evidence_items = _build_guidance_evidence_items(evidence_items)

        merged_fetched_rows = [_page_to_jsonable_row(page) for page in merged_fetched_pages]
        external_fetched_trace_rows = self._build_fetched_page_trace_rows(external_fetched_pages)
        selected_search_fetched_trace_rows = self._build_fetched_page_trace_rows(selected_search_fetched_pages)
        merged_fetched_trace_rows = self._build_fetched_page_trace_rows(merged_fetched_pages)
        elapsed = round(time.time() - started_at, 2)
        self._progress(f"TOPIC_END topic_id={topic_id} elapsed_seconds={elapsed}")
        return {
            "topic_id": topic_id,
            "guidance": {
                "topic_id": topic_id,
                "query": query,
                "answer": answer,
                "topic_guidance": topic_brief,
                "evidence_items": guidance_evidence_items,
                "answer_assessment": answer_assessment,
            },
            "merged_fetched_rows": merged_fetched_rows,
            "trace": {
                "topic_id": topic_id,
                "query": query,
                "answer": answer,
                "claims": claims,
                "search_queries": as_jsonable_rows(search_queries),
                "raw_search_candidates": as_jsonable_rows(raw_search_candidates),
                "search_candidates_for_selection": as_jsonable_rows(search_candidates_for_selection),
                "selected_search_urls": as_jsonable_rows(selected_search_urls),
                "external_fetched_pages": external_fetched_trace_rows,
                "selected_search_fetched_pages": selected_search_fetched_trace_rows,
                "merged_fetched_pages": merged_fetched_trace_rows,
                "evidence_items": evidence_items,
                "topic_brief": topic_brief,
                "answer_assessment": answer_assessment,
            },
        }

    def _build_failed_topic_row(self, *, topic_id: str, error_text: str) -> dict[str, Any]:
        return {
            "topic_id": topic_id,
            "guidance": {
                "topic_id": topic_id,
                "query": "",
                "answer": "",
                "topic_guidance": "",
                "evidence_items": [],
                "answer_assessment": {},
            },
            "merged_fetched_rows": [],
            "trace": {
                "topic_id": topic_id,
                "query": "",
                "answer": "",
                "claims": [],
                "search_queries": [],
                "raw_search_candidates": [],
                "search_candidates_for_selection": [],
                "selected_search_urls": [],
                "external_fetched_pages": [],
                "selected_search_fetched_pages": [],
                "merged_fetched_pages": [],
                "evidence_items": [],
                "topic_brief": "",
                "answer_assessment": {},
                "error": error_text,
            },
        }

    def _progress(self, message: str) -> None:
        if self.config.print_progress:
            print(f"[prefetch-answer] {message}", flush=True)

    def _initialize_output_files(self, paths: list[Path]) -> None:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(path, [])

    def _append_jsonl_row(self, path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_jsonl_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("a", encoding="utf-8") as file_handle:
            for row in rows:
                file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _append_completed_topic_row(
        self,
        *,
        guidance_path: Path,
        fetched_path: Path,
        trace_path: Path,
        row: dict[str, Any],
    ) -> None:
        self._append_jsonl_row(guidance_path, row["guidance"])
        self._append_jsonl_rows(fetched_path, row["merged_fetched_rows"])
        self._append_jsonl_row(trace_path, row["trace"])

    def _build_fetched_page_trace_rows(self, pages: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        max_chars = int(self.config.stage_config.max_page_text_chars)
        for page in pages:
            full_text = str(getattr(page, "full_text", "") or "")
            text_len = len(full_text)
            out.append(
                {
                    "topic_id": str(getattr(page, "topic_id", "") or ""),
                    "query": str(getattr(page, "query", "") or ""),
                    "url_id": str(getattr(page, "url_id", "") or ""),
                    "source_url": str(getattr(page, "source_url", "") or ""),
                    "source_title": str(getattr(page, "source_title", "") or ""),
                    "source_type": str(getattr(page, "source_type", "") or ""),
                    "candidate_ids": list(getattr(page, "candidate_ids", []) or []),
                    "query_ids": list(getattr(page, "query_ids", []) or []),
                    "fetch_error": getattr(page, "fetch_error", None),
                    "text_char_len": text_len,
                    "is_truncated": text_len >= max_chars if text_len > 0 else False,
                    "max_page_text_chars": max_chars,
                }
            )
        return out


def main() -> None:
    root = _project_root()
    input_jsonl = root / INPUT_JSONL
    output_dir = root / OUTPUT_DIR
    _progress("BOOT run_prefetch_topic_evidence_with_answer.py")
    _progress(f"PATHS input_jsonl={input_jsonl} output_dir={output_dir}")

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    _prepare_output_dir_and_check_conflicts(output_dir, OUTPUT_CONFLICT_POLICY)

    reused_fetched_pages_by_topic: dict[str, list[Any]] = {}
    reused_path_str = str(REUSE_FETCHED_FULLTEXT_JSONL or "").strip()
    if reused_path_str:
        reused_path = root / reused_path_str
        if not reused_path.exists():
            raise FileNotFoundError(f"Reused fetched_fulltext JSONL not found: {reused_path}")
        reused_fetched_pages_by_topic = _load_reused_fetched_pages(reused_path)
        _progress(
            f"REUSE_FETCHED enabled path={reused_path} topic_count={len(reused_fetched_pages_by_topic)}"
        )

    topic_rows = _read_topic_rows(input_jsonl, use_input_urls=bool(USE_INPUT_URLS))
    topic_rows = topic_rows[max(0, int(TOPIC_START_INDEX)) :]
    if LIMIT_TOPICS is not None:
        topic_rows = topic_rows[: max(0, int(LIMIT_TOPICS))]
    _progress(
        f"INPUT_LOADED topic_count={len(topic_rows)} use_input_urls={USE_INPUT_URLS} "
        f"start_index={max(0, int(TOPIC_START_INDEX))}"
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

    pipeline = PrefetchTopicEvidenceWithAnswerPipeline(
        config=config,
        output_dir=output_dir,
        reused_fetched_pages_by_topic=reused_fetched_pages_by_topic,
    )
    report = pipeline.run(topic_rows=topic_rows)

    report_path = output_dir / "run_prefetch_topic_evidence_with_answer_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _progress(f"REPORT_WRITTEN path={report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
