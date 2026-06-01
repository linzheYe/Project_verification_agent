from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from scripts.data_io import write_jsonl
from scripts.llm_api import LLMClient
from scripts.prefetch_topic_evidence_stages import (
    PrefetchStageConfig,
    as_jsonable_rows,
    merge_fetched_pages_external_first,
    prepare_search_candidates_for_selection,
    stage1_plan_and_search,
    stage2_prepare_external_fulltext,
    stage3_select_search_urls,
    stage4_fetch_selected_search_fulltext,
    stage5_extract_evidence,
)


@dataclass(frozen=True)
class PrefetchPipelineConfig:
    llm_model: str
    enable_topic_parallel: bool = False
    topic_parallel_workers: int = 4
    print_progress: bool = True
    stage_config: PrefetchStageConfig = PrefetchStageConfig()


class PrefetchTopicEvidencePipeline:
    """Readable topic prefetch pipeline with trusted external URL support."""

    def __init__(self, *, config: PrefetchPipelineConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, *, topic_rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full 5-stage flow for all topics and write final outputs."""
        topic_inputs = self._build_topic_inputs(topic_rows)
        total_topics = len(topic_inputs)
        self._progress(f"START total_topics={total_topics} parallel={self.config.enable_topic_parallel}")

        grounding_path = self.output_dir / "topic_grounding.jsonl"
        guidance_path = self.output_dir / "topic_guidance.jsonl"
        fetched_path = self.output_dir / "fetched_fulltext_pages.jsonl"
        trace_path = self.output_dir / "topic_grounding_trace.jsonl"
        self._initialize_output_files([grounding_path, guidance_path, fetched_path, trace_path])

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
                            grounding_path=grounding_path,
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
                            grounding_path=grounding_path,
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
                        grounding_path=grounding_path,
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
                        grounding_path=grounding_path,
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
            "topic_grounding_path": str(grounding_path),
            "topic_guidance_path": str(guidance_path),
            "fetched_fulltext_path": str(fetched_path),
            "topic_grounding_trace_path": str(trace_path),
        }

    def _build_topic_inputs(self, topic_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize user-provided topic rows into pipeline input schema."""
        topic_inputs: list[dict[str, Any]] = []
        clean_rows = [row for row in topic_rows if str(row.get("query") or "").strip()]
        for topic_index, row in enumerate(clean_rows, start=1):
            query = str(row.get("query") or "").strip()
            topic_id = str(row.get("topic_id") or "").strip()
            if not topic_id:
                topic_id = f"T{topic_index}"
            topic_inputs.append(
                {
                    "topic_id": topic_id,
                    "query": query,
                    "claims": [{"claim_id": f"{topic_id}_C1", "claim_text": query}],
                    "external_urls": [str(url).strip() for url in (row.get("external_urls") or []) if str(url).strip()],
                }
            )
        return topic_inputs

    def _process_single_topic(self, topic: dict[str, Any]) -> dict[str, Any]:
        """Process one topic end-to-end and return rows ready for output files."""
        started_at = time.time()
        topic_id = topic["topic_id"]
        query = topic["query"]
        claims = topic["claims"]
        external_urls = topic["external_urls"]
        self._progress(f"TOPIC_START topic_id={topic_id}")
        llm = LLMClient(provider="openrouter", model=self.config.llm_model, timeout_seconds=60, max_retries=1)

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
        evidence_items, topic_brief = stage5_extract_evidence(
            llm=llm,
            topic_id=topic_id,
            query=query,
            claims=claims,
            merged_fetched_pages=merged_fetched_pages,
        )

        merged_fetched_rows = as_jsonable_rows(merged_fetched_pages)
        external_fetched_trace_rows = self._build_fetched_page_trace_rows(external_fetched_pages)
        selected_search_fetched_trace_rows = self._build_fetched_page_trace_rows(selected_search_fetched_pages)
        merged_fetched_trace_rows = self._build_fetched_page_trace_rows(merged_fetched_pages)
        elapsed = round(time.time() - started_at, 2)
        self._progress(f"TOPIC_END topic_id={topic_id} elapsed_seconds={elapsed}")
        return {
            "topic_id": topic_id,
            "grounding": {
                "topic_id": topic_id,
                "query": query,
                "evidence_items": evidence_items,
                "topic_brief": topic_brief,
            },
            "guidance": {
                "topic_id": topic_id,
                "query": query,
                "topic_guidance": topic_brief,
            },
            "merged_fetched_rows": merged_fetched_rows,
            "trace": {
                "topic_id": topic_id,
                "query": query,
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
            },
        }

    def _build_failed_topic_row(self, *, topic_id: str, error_text: str) -> dict[str, Any]:
        """Build output row when one topic fails; keeps global run alive."""
        return {
            "topic_id": topic_id,
            "grounding": {
                "topic_id": topic_id,
                "query": "",
                "evidence_items": [],
                "topic_brief": "",
            },
            "guidance": {
                "topic_id": topic_id,
                "query": "",
                "topic_guidance": "",
            },
            "merged_fetched_rows": [],
            "trace": {
                "topic_id": topic_id,
                "query": "",
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
                "error": error_text,
            },
        }

    def _progress(self, message: str) -> None:
        if self.config.print_progress:
            print(f"[prefetch] {message}", flush=True)

    def _initialize_output_files(self, paths: list[Path]) -> None:
        """Create/clear output files so we can append rows as topics complete."""
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
        grounding_path: Path,
        guidance_path: Path,
        fetched_path: Path,
        trace_path: Path,
        row: dict[str, Any],
    ) -> None:
        """Append one completed topic result to all output files immediately."""
        self._append_jsonl_row(grounding_path, row["grounding"])
        self._append_jsonl_row(guidance_path, row["guidance"])
        self._append_jsonl_rows(fetched_path, row["merged_fetched_rows"])
        self._append_jsonl_row(trace_path, row["trace"])

    def _build_fetched_page_trace_rows(self, pages: list[Any]) -> list[dict[str, Any]]:
        """Build lightweight fetched-page trace rows without full_text payload."""
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
