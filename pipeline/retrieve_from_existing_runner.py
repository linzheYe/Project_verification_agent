from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.pipeline_evidence_retrieve import EvidenceRetrievePipeline
from scripts.data_io import read_jsonl, write_jsonl
from scripts.llm_api import LLMClient


def _topic_key_by_query(row: dict) -> str:
    sample = row.get("sample") or {}
    return str(sample.get("query") or "").strip()


def _topic_slug(query: str) -> str:
    digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:10]
    return f"query_{digest}"


def _truncate_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _append_jsonl_file(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out, input_path.open("r", encoding="utf-8") as inp:
        for line in inp:
            if line.strip():
                out.write(line)


def _sum_timing_json(paths: list[Path], output_path: Path) -> None:
    timing_total: dict[str, float] = {}
    counts_total: dict[str, int] = {}
    for path in paths:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        for k, v in (obj.get("timing_seconds") or {}).items():
            timing_total[k] = float(timing_total.get(k, 0.0)) + float(v or 0.0)
        for k, v in (obj.get("counts") or {}).items():
            counts_total[k] = int(counts_total.get(k, 0)) + int(v or 0)
    output_path.write_text(
        json.dumps(
            {
                "timing_seconds": dict(sorted(timing_total.items(), key=lambda x: x[0])),
                "counts": dict(sorted(counts_total.items(), key=lambda x: x[0])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_topic_worker(
    *,
    out_dir: Path,
    query: str,
    split_rows: list[dict],
    clean_rows: list[dict],
    retrieve_model: str,
    max_retrieval_round: int,
    max_recoverable_retries_per_round: int,
    max_must_answer_retries: int,
    max_snippet_per_search: int,
    max_initial_snippets: int,
    enable_bm25_initial_select: bool,
    duplicate_threshold: float,
    no_result_defer_threshold: int,
    no_progress_defer_threshold: int,
    top_k_freq_snippets: int,
    query_cache_jaccard_threshold: float,
    topic_grounding_path: Path | None = None,
    topic_guidance_path: Path | None = None,
) -> dict[str, str]:
    topic_dir = out_dir / "topics" / _topic_slug(query)
    topic_dir.mkdir(parents=True, exist_ok=True)

    split_topic_path = topic_dir / "split_claims.topic.jsonl"
    clean_topic_path = topic_dir / "cleaned_answer.topic.jsonl"
    retrieval_trace_path = topic_dir / "retrieval_trace.topic.jsonl"
    claim_status_path = topic_dir / "claim_snippets_status.topic.jsonl"
    question_pool_path = topic_dir / "question_evidence_pool.topic.jsonl"
    evidence_results_path = topic_dir / "evidence_results.topic.jsonl"
    high_freq_snippets_path = topic_dir / "high_freq_snippets.topic.jsonl"
    timing_summary_path = topic_dir / "timing_summary.topic.json"

    write_jsonl(split_topic_path, split_rows)
    write_jsonl(clean_topic_path, clean_rows)

    pipeline = EvidenceRetrievePipeline(
        split_path=split_topic_path,
        clean_path=clean_topic_path,
        retrieval_trace_path=retrieval_trace_path,
        claim_status_path=claim_status_path,
        question_pool_path=question_pool_path,
        evidence_results_path=evidence_results_path,
        high_freq_snippets_path=high_freq_snippets_path,
        timing_summary_path=timing_summary_path,
        llm_client=LLMClient(provider="openrouter", model=retrieve_model, timeout_seconds=45, max_retries=1),
        max_retrieval_round=max_retrieval_round,
        max_recoverable_retries_per_round=max_recoverable_retries_per_round,
        max_must_answer_retries=max_must_answer_retries,
        max_snippet_per_search=max_snippet_per_search,
        max_initial_snippets=max_initial_snippets,
        enable_bm25_initial_select=enable_bm25_initial_select,
        duplicate_threshold=duplicate_threshold,
        no_result_defer_threshold=no_result_defer_threshold,
        no_progress_defer_threshold=no_progress_defer_threshold,
        top_k_freq_snippets=top_k_freq_snippets,
        query_cache_jaccard_threshold=query_cache_jaccard_threshold,
        topic_grounding_path=topic_grounding_path,
        topic_guidance_path=topic_guidance_path,
    )
    pipeline.run()

    return {
        "query": query,
        "retrieval_trace_path": str(retrieval_trace_path),
        "claim_status_path": str(claim_status_path),
        "question_pool_path": str(question_pool_path),
        "evidence_results_path": str(evidence_results_path),
        "high_freq_snippets_path": str(high_freq_snippets_path),
        "timing_summary_path": str(timing_summary_path),
    }


def run_retrieval_from_existing_files(
    *,
    split_path: Path,
    clean_path: Path,
    out_dir: Path,
    retrieve_model: str,
    max_retrieval_round: int,
    max_recoverable_retries_per_round: int,
    max_must_answer_retries: int,
    max_snippet_per_search: int,
    max_initial_snippets: int,
    enable_bm25_initial_select: bool,
    duplicate_threshold: float,
    no_result_defer_threshold: int,
    no_progress_defer_threshold: int,
    top_k_freq_snippets: int,
    query_cache_jaccard_threshold: float,
    enable_topic_parallel: bool,
    max_topic_workers: int,
    topic_grounding_path: Path | None = None,
    topic_guidance_path: Path | None = None,
) -> dict[str, str | int | bool]:
    retrieval_trace_path = out_dir / "retrieval_trace.jsonl"
    claim_status_path = out_dir / "claim_snippets_status.jsonl"
    question_pool_path = out_dir / "question_evidence_pool.jsonl"
    evidence_results_path = out_dir / "evidence_results.jsonl"
    high_freq_snippets_path = out_dir / "high_freq_snippets.jsonl"
    timing_summary_path = out_dir / "timing_summary.json"

    split_all = read_jsonl(split_path)
    clean_all = read_jsonl(clean_path)
    split_by_topic: dict[str, list[dict]] = {}
    clean_by_topic: dict[str, list[dict]] = {}
    for row in split_all:
        key = _topic_key_by_query(row)
        if key:
            split_by_topic.setdefault(key, []).append(row)
    for row in clean_all:
        key = _topic_key_by_query(row)
        if key:
            clean_by_topic.setdefault(key, []).append(row)

    topics = sorted(split_by_topic.keys())
    if not topics:
        raise RuntimeError("no topics found in split_path")

    _truncate_jsonl(retrieval_trace_path)
    _truncate_jsonl(claim_status_path)
    _truncate_jsonl(question_pool_path)
    _truncate_jsonl(evidence_results_path)
    _truncate_jsonl(high_freq_snippets_path)

    worker_outputs: list[dict[str, str]] = []
    if enable_topic_parallel:
        with ThreadPoolExecutor(max_workers=max_topic_workers) as ex:
            fut_to_topic = {}
            for query in topics:
                fut = ex.submit(
                    _run_topic_worker,
                    out_dir=out_dir,
                    query=query,
                    split_rows=split_by_topic[query],
                    clean_rows=clean_by_topic.get(query, []),
                    retrieve_model=retrieve_model,
                    max_retrieval_round=max_retrieval_round,
                    max_recoverable_retries_per_round=max_recoverable_retries_per_round,
                    max_must_answer_retries=max_must_answer_retries,
                    max_snippet_per_search=max_snippet_per_search,
                    max_initial_snippets=max_initial_snippets,
                    enable_bm25_initial_select=enable_bm25_initial_select,
                    duplicate_threshold=duplicate_threshold,
                    no_result_defer_threshold=no_result_defer_threshold,
                    no_progress_defer_threshold=no_progress_defer_threshold,
                    top_k_freq_snippets=top_k_freq_snippets,
                    query_cache_jaccard_threshold=query_cache_jaccard_threshold,
                    topic_grounding_path=topic_grounding_path,
                    topic_guidance_path=topic_guidance_path,
                )
                fut_to_topic[fut] = query
            for fut in as_completed(fut_to_topic):
                worker_outputs.append(fut.result())
    else:
        for query in topics:
            worker_outputs.append(
                _run_topic_worker(
                    out_dir=out_dir,
                    query=query,
                    split_rows=split_by_topic[query],
                    clean_rows=clean_by_topic.get(query, []),
                    retrieve_model=retrieve_model,
                    max_retrieval_round=max_retrieval_round,
                    max_recoverable_retries_per_round=max_recoverable_retries_per_round,
                    max_must_answer_retries=max_must_answer_retries,
                    max_snippet_per_search=max_snippet_per_search,
                    max_initial_snippets=max_initial_snippets,
                    enable_bm25_initial_select=enable_bm25_initial_select,
                    duplicate_threshold=duplicate_threshold,
                    no_result_defer_threshold=no_result_defer_threshold,
                    no_progress_defer_threshold=no_progress_defer_threshold,
                    top_k_freq_snippets=top_k_freq_snippets,
                    query_cache_jaccard_threshold=query_cache_jaccard_threshold,
                    topic_grounding_path=topic_grounding_path,
                    topic_guidance_path=topic_guidance_path,
                )
            )

    timing_paths: list[Path] = []
    for out in worker_outputs:
        _append_jsonl_file(Path(out["retrieval_trace_path"]), retrieval_trace_path)
        _append_jsonl_file(Path(out["claim_status_path"]), claim_status_path)
        _append_jsonl_file(Path(out["question_pool_path"]), question_pool_path)
        _append_jsonl_file(Path(out["evidence_results_path"]), evidence_results_path)
        _append_jsonl_file(Path(out["high_freq_snippets_path"]), high_freq_snippets_path)
        timing_paths.append(Path(out["timing_summary_path"]))
    _sum_timing_json(timing_paths, timing_summary_path)

    return {
        "topic_count": len(topics),
        "enable_topic_parallel": enable_topic_parallel,
        "max_topic_workers": max_topic_workers,
        "topics_output_dir": str(out_dir / "topics"),
        "retrieval_trace_path": str(retrieval_trace_path),
        "claim_status_path": str(claim_status_path),
        "question_pool_path": str(question_pool_path),
        "evidence_results_path": str(evidence_results_path),
        "high_freq_snippets_path": str(high_freq_snippets_path),
        "timing_summary_path": str(timing_summary_path),
    }
