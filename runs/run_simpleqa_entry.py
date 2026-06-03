#!/usr/bin/env python3
from __future__ import annotations

"""
Single source of truth for SimpleQA run parameters.

Parameter ownership:
- Edit business parameters ONLY in this file.
- Batch launcher should only pass which input file to run, never duplicate model/stage thresholds.
"""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_prepare_claims import PrepareClaimsPipeline
from pipeline.retrieve_from_existing_runner import run_retrieval_from_existing_files
from pipeline.simpleqa_run_config import STAGE_RANK, SimpleQARunConfig, resolve_output_dir, validate_paths, validate_stage_control
from scripts.data_io import build_source_records, read_jsonl, write_jsonl
from scripts.ddg_search import ensure_ddgs_dependency
from scripts.llm_api import LLMClient

# =========================
# [ONLY PLACE TO EDIT BUSINESS PARAMS]
# =========================
INPUT_MODE = "model_jsonl"  # 输入文件格式: 仅支持 model_jsonl；rubrics请用 run_simpleqa_rubrics_from_evidence.py
INPUT_PATH = Path("response_data/benchmark_10_5/simpleqa_model_answers_10_5.jsonl")  # 输入文件路径(相对项目根目录)
# INPUT_PATH = Path("response_data/collected_30_5/claude-sonnet-4.6_30_5.jsonl")  # 输入文件路径(相对项目根目录)
# INPUT_PATH = Path("response_data/chinese10_english10/bilingual_benchmark_data.jsonl") 
START_LINE = 1 # 从输入文件第几行开始跑(1-based)
NUM_LINES = 10 # 本次最多跑多少行
OUTPUT_BASE_DIR = Path("clean_split_retrive_data/try")  # 输出根目录(每次run会在这里建一个子目录)
RUN_NAME = "try32_small"  # 本次run目录名; 留空则自动用输入文件名(去后缀)

# 如果要恢复上一次跑到一半的任务，保持同一个 RUN_NAME 直接再跑即可
# 如果想强制全量重跑，把 RETRY_ONLY_ERRORS = False，或者换一个新的 RUN_NAME

# 输入字段映射:
# - id: original_index -> id，若都缺失则回退 simpleqa_{idx}
# - query: query -> original_question
# - model: model -> actual_model_used，若都缺失则 unknown_model
# - model_answer: response -> model_answer -> answer
INPUT_FIELD_CANDIDATES = {
    "id": ("original_index", "id"),
    "query": ("query", "original_question"),
    "model": ("model", "actual_model_used"),
    "model_answer": ("response", "model_answer", "answer"),
}

RUN_TARGET_STAGE = "retrieve"  # clean|split|retrieve
REUSE_FROM_STAGE = "split"  # none|clean|split|retrieve
RETRY_ONLY_ERRORS = True  # 若已有 clean/split 输出，则只重跑 error 项并回填

EXISTING_TOPIC_GROUNDING_PATH = Path("prefetch_data/topics_10/output_with_urls_6/topic_grounding.jsonl")
EXISTING_TOPIC_GUIDANCE_PATH = Path("prefetch_data/topics_10/output_with_urls_6/topic_guidance.jsonl")

EXISTING_CLEAN_PATH = Path("clean_split_retrive_data/try/try29_small/cleaned_answer.jsonl")
EXISTING_SPLIT_PATH = Path("clean_split_retrive_data/try/try29_small/split_claims.jsonl")


# CLEAN_MODEL = "google/gemini-3.1-flash-lite-preview"
CLEAN_MODEL = "openai/gpt-5.4-nano"
SPLIT_MODEL = "google/gemini-3.1-flash-lite-preview"
RETRIEVE_MODEL = "openai/gpt-5.4-nano"

ENABLE_PREPARE_PARALLEL = True
MAX_PREPARE_WORKERS = 5 #workers

MAX_RETRIEVAL_ROUND = 3
MAX_RECOVERABLE_RETRIES_PER_ROUND = 2
MAX_MUST_ANSWER_RETRIES = 2
MAX_SNIPPET_PER_SEARCH = 10
MAX_INITIAL_SNIPPETS = 8
ENABLE_BM25_INITIAL_SELECT = True
DUPLICATE_THRESHOLD = 0.9
NO_RESULT_DEFER_THRESHOLD = 1
NO_PROGRESS_DEFER_THRESHOLD = 1
TOP_K_FREQ_SNIPPETS = 8
QUERY_CACHE_JACCARD_THRESHOLD = 0.8
ENABLE_TOPIC_PARALLEL = True
MAX_TOPIC_WORKERS = 8 #workers


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sample_row_key(row: dict[str, Any]) -> tuple[str, str]:
    sample = row.get("sample") or {}
    return str(sample.get("id") or "").strip(), str(sample.get("query") or "").strip()


def _model_item_key(item: dict[str, Any]) -> int:
    return int(item.get("model_sequence_id") or 0)


def _count_stage_errors(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        for item in (row.get("model_outputs") or {}).values():
            if str((item or {}).get("error") or "").strip():
                total += 1
    return total


def _needs_clean_retry(rec: dict[str, Any]) -> bool:
    for _, item in (rec.get("model_outputs") or {}).items():
        if str((item or {}).get("error") or "").strip():
            return True
        if not str((item or {}).get("cleaned_model_answer") or "").strip():
            return True
    return False


def _needs_split_retry(rec: dict[str, Any]) -> bool:
    for _, item in (rec.get("model_outputs") or {}).items():
        if str((item or {}).get("error") or "").strip():
            return True
        if not (item or {}).get("atomic_claims"):
            return True
    return False


def _merge_stage_rows(base_rows: list[dict[str, Any]], patched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patch_by_key = {_sample_row_key(row): row for row in patched_rows}
    merged: list[dict[str, Any]] = []
    for row in base_rows:
        key = _sample_row_key(row)
        patch = patch_by_key.get(key)
        if not patch:
            merged.append(row)
            continue
        merged_outputs = dict(row.get("model_outputs") or {})
        for seq_key, item in (patch.get("model_outputs") or {}).items():
            merged_outputs[str(seq_key)] = item
        merged.append({"sample": row.get("sample"), "model_outputs": merged_outputs})
    return merged


def _project_clean_retry_records(source_records: list[dict[str, Any]], clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_by_key = {_sample_row_key(row): row for row in clean_rows}
    retry_records: list[dict[str, Any]] = []
    for rec in source_records:
        key = _sample_row_key(rec)
        existing = existing_by_key.get(key)
        if not existing:
            retry_records.append(rec)
            continue
        retry_outputs: dict[str, dict[str, Any]] = {}
        existing_outputs_by_seq = {
            _model_item_key(item): item for item in (existing.get("model_outputs") or {}).values()
        }
        for seq_key, item in (rec.get("model_outputs") or {}).items():
            seq = _model_item_key(item)
            existing_item = existing_outputs_by_seq.get(seq) or {}
            if str(existing_item.get("error") or "").strip() or not str(existing_item.get("cleaned_model_answer") or "").strip():
                retry_outputs[str(seq_key)] = item
        if retry_outputs:
            retry_records.append({"sample": rec.get("sample"), "model_outputs": retry_outputs})
    return retry_records


def _project_split_retry_rows(clean_rows: list[dict[str, Any]], split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_by_key = {_sample_row_key(row): row for row in split_rows}
    retry_rows: list[dict[str, Any]] = []
    for clean_row in clean_rows:
        key = _sample_row_key(clean_row)
        existing = existing_by_key.get(key)
        if not existing:
            retry_rows.append(clean_row)
            continue
        retry_outputs: dict[str, dict[str, Any]] = {}
        existing_outputs_by_seq = {
            _model_item_key(item): item for item in (existing.get("model_outputs") or {}).values()
        }
        for seq_key, item in (clean_row.get("model_outputs") or {}).items():
            seq = _model_item_key(item)
            existing_item = existing_outputs_by_seq.get(seq) or {}
            if str(existing_item.get("error") or "").strip() or not (existing_item.get("atomic_claims") or []):
                retry_outputs[str(seq_key)] = item
        if retry_outputs:
            retry_rows.append({"sample": clean_row.get("sample"), "model_outputs": retry_outputs})
    return retry_rows


def _load_config() -> SimpleQARunConfig:
    input_path = Path(os.environ.get("SIMPLEQA_INPUT_MODEL_JSONL", str(PROJECT_ROOT / INPUT_PATH)))
    run_name = os.environ.get("SIMPLEQA_RUN_NAME_OVERRIDE", RUN_NAME).strip()
    return SimpleQARunConfig(
        input_mode=INPUT_MODE,
        input_path=input_path,
        output_base_dir=PROJECT_ROOT / OUTPUT_BASE_DIR,
        run_name=run_name,
        run_target_stage=RUN_TARGET_STAGE,
        reuse_from_stage=REUSE_FROM_STAGE,
        existing_clean_path=PROJECT_ROOT / EXISTING_CLEAN_PATH,
        existing_split_path=PROJECT_ROOT / EXISTING_SPLIT_PATH,
        existing_topic_grounding_path=PROJECT_ROOT / EXISTING_TOPIC_GROUNDING_PATH,
        existing_topic_guidance_path=PROJECT_ROOT / EXISTING_TOPIC_GUIDANCE_PATH,
        existing_evidence_results_path=None,
        clean_model=CLEAN_MODEL,
        split_model=SPLIT_MODEL,
        prefetch_model="",
        retrieve_model=RETRIEVE_MODEL,
        enable_prepare_parallel=ENABLE_PREPARE_PARALLEL,
        max_prepare_workers=MAX_PREPARE_WORKERS,
        prefetch_enable_topic_parallel=False,
        prefetch_max_topic_workers=1,
        prefetch_wiki_query_count=3,
        prefetch_web_query_count=3,
        prefetch_snippets_per_query=5,
        prefetch_max_selected_candidates=6,
        prefetch_fetch_timeout_seconds=20,
        prefetch_max_page_text_chars=15000,
        prefetch_near_dup_threshold=0.9,
        max_retrieval_round=MAX_RETRIEVAL_ROUND,
        max_recoverable_retries_per_round=MAX_RECOVERABLE_RETRIES_PER_ROUND,
        max_must_answer_retries=MAX_MUST_ANSWER_RETRIES,
        max_snippet_per_search=MAX_SNIPPET_PER_SEARCH,
        max_initial_snippets=MAX_INITIAL_SNIPPETS,
        enable_bm25_initial_select=ENABLE_BM25_INITIAL_SELECT,
        duplicate_threshold=DUPLICATE_THRESHOLD,
        no_result_defer_threshold=NO_RESULT_DEFER_THRESHOLD,
        no_progress_defer_threshold=NO_PROGRESS_DEFER_THRESHOLD,
        top_k_freq_snippets=TOP_K_FREQ_SNIPPETS,
        query_cache_jaccard_threshold=QUERY_CACHE_JACCARD_THRESHOLD,
        enable_topic_parallel=ENABLE_TOPIC_PARALLEL,
        max_topic_workers=MAX_TOPIC_WORKERS,
    )


def _validate_entry_stage_policy(cfg: SimpleQARunConfig) -> None:
    if cfg.run_target_stage == "prefetch":
        raise ValueError("RUN_TARGET_STAGE='prefetch' is disabled. Prefetch must be prepared in advance.")
    if cfg.run_target_stage == "rubrics":
        raise ValueError("RUN_TARGET_STAGE='rubrics' is disabled in this script. Use run_simpleqa_rubrics_from_evidence.py.")
    if cfg.reuse_from_stage == "prefetch":
        raise ValueError("REUSE_FROM_STAGE='prefetch' is disabled in this entry script.")


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._-") or "run"


def _build_source_rows(cfg: SimpleQARunConfig) -> list[dict[str, Any]]:
    if cfg.input_mode != "model_jsonl":
        raise ValueError(
            "unsupported INPUT_MODE for run_simpleqa_entry.py. "
            "Only 'model_jsonl' is supported; use run_simpleqa_rubrics_from_evidence.py for rubrics."
        )

    rows = read_jsonl(cfg.input_path)
    s = max(0, int(START_LINE) - 1)
    e = s + max(0, int(NUM_LINES))
    rows = rows[s:e]
    if not rows:
        raise RuntimeError("input jsonl is empty")

    def _pick(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            val = row.get(key)
            if val is not None and str(val).strip():
                return val
        return None

    out: list[dict[str, Any]] = []
    invalid_rows: list[tuple[int, str]] = []
    for idx, row in enumerate(rows, start=1):
        row_id = _pick(row, tuple(INPUT_FIELD_CANDIDATES["id"]))
        query = str(_pick(row, tuple(INPUT_FIELD_CANDIDATES["query"])) or "").strip()
        model = str(_pick(row, tuple(INPUT_FIELD_CANDIDATES["model"])) or "unknown_model").strip()
        model_answer = str(_pick(row, tuple(INPUT_FIELD_CANDIDATES["model_answer"])) or "").strip()
        sid = str(row_id).strip() if row_id is not None else ""

        if not sid:
            sid = f"simpleqa_{idx}"

        missing: list[str] = []
        if not query:
            missing.append("query")
        if not model_answer:
            missing.append("model_answer")
        if missing:
            invalid_rows.append((idx, ",".join(missing)))
            continue

        out.append(
            {
                "id": sid,
                "query": query,
                "model": model,
                "model_answer": model_answer,
            }
        )

    if not out:
        raise RuntimeError(
            "no valid input rows after field mapping. "
            f"required semantic fields: query({INPUT_FIELD_CANDIDATES['query']}), "
            f"model_answer({INPUT_FIELD_CANDIDATES['model_answer']})."
        )
    if invalid_rows:
        preview = "; ".join(f"line#{ln} missing={miss}" for ln, miss in invalid_rows[:5])
        raise RuntimeError(
            f"invalid input rows detected: {len(invalid_rows)} rows missing required fields. "
            f"examples: {preview}. "
            f"query candidates={INPUT_FIELD_CANDIDATES['query']}, "
            f"model_answer candidates={INPUT_FIELD_CANDIDATES['model_answer']}."
        )
    return out


def _run_clean_records(records: list[dict[str, Any]], *, model: str, workers: int, out_dir: Path, parallel: bool) -> list[dict[str, Any]]:
    def run_one(rec: dict[str, Any]) -> dict[str, Any]:
        sample = rec.get("sample") or {}
        sample_id = str(sample.get("id") or "").strip()
        query = str(sample.get("query") or "").strip()
        t0 = time.perf_counter()
        print(f"[{_ts()}] [clean:start] sample_id={sample_id} query={query[:120]!r} model={model}", flush=True)
        llm = LLMClient(provider="openrouter", model=model, timeout_seconds=45, max_retries=1)
        pipe = PrepareClaimsPipeline(
            source_path=out_dir / "unused.source.jsonl",
            clean_path=out_dir / "unused.clean.jsonl",
            split_path=out_dir / "unused.split.jsonl",
            group_path=out_dir / "unused.group.jsonl",
            llm_client=llm,
        )
        row = pipe.run_clean_stage([rec])[0]
        elapsed = time.perf_counter() - t0
        error_count = sum(
            1
            for _, item in ((row.get("model_outputs") or {}).items())
            if str((item or {}).get("error") or "").strip()
        )
        print(
            f"[{_ts()}] [clean:done] sample_id={sample_id} elapsed={elapsed:.2f}s errors={error_count}",
            flush=True,
        )
        return row

    if not parallel:
        print(
            f"[{_ts()}] [clean] mode=serial records={len(records)} model={model} timeout=45 retries=1",
            flush=True,
        )
        llm = LLMClient(provider="openrouter", model=model, timeout_seconds=45, max_retries=1)
        pipe = PrepareClaimsPipeline(
            source_path=out_dir / "unused.source.jsonl",
            clean_path=out_dir / "unused.clean.jsonl",
            split_path=out_dir / "unused.split.jsonl",
            group_path=out_dir / "unused.group.jsonl",
            llm_client=llm,
        )
        return pipe.run_clean_stage(records)

    print(
        f"[{_ts()}] [clean] mode=parallel records={len(records)} workers={workers} model={model} timeout=45 retries=1",
        flush=True,
    )
    out: list[dict[str, Any] | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(run_one, rec): i for i, rec in enumerate(records)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                out[idx] = fut.result()
            except Exception as e:
                rec = records[idx]
                sample = rec.get("sample") or {}
                sample_id = str(sample.get("id") or "").strip()
                print(f"[{_ts()}] [clean:error] idx={idx} sample_id={sample_id} error={e}", flush=True)
                fallback_outputs: dict[str, dict[str, Any]] = {}
                for seq_key, item in (rec.get("model_outputs") or {}).items():
                    fallback_outputs[str(seq_key)] = {
                        "model_sequence_id": int((item or {}).get("model_sequence_id") or 0),
                        "model_name": str((item or {}).get("model_name") or ""),
                        "cleaned_model_answer": "",
                        "error": f"clean_runner_error: {e}",
                    }
                out[idx] = {"sample": rec.get("sample"), "model_outputs": fallback_outputs}
    return [x for x in out if x is not None]


def _run_split_rows(clean_rows: list[dict[str, Any]], *, model: str, workers: int, out_dir: Path, parallel: bool) -> list[dict[str, Any]]:
    def run_one(clean_row: dict[str, Any]) -> dict[str, Any]:
        sample = clean_row.get("sample") or {}
        sample_id = str(sample.get("id") or "").strip()
        query = str(sample.get("query") or "").strip()
        t0 = time.perf_counter()
        print(f"[{_ts()}] [split:start] sample_id={sample_id} query={query[:120]!r} model={model}", flush=True)
        llm = LLMClient(provider="openrouter", model=model, timeout_seconds=45, max_retries=1)
        pipe = PrepareClaimsPipeline(
            source_path=out_dir / "unused.source.jsonl",
            clean_path=out_dir / "unused.clean.jsonl",
            split_path=out_dir / "unused.split.jsonl",
            group_path=out_dir / "unused.group.jsonl",
            llm_client=llm,
        )
        row = pipe.run_split_stage([clean_row])[0]
        elapsed = time.perf_counter() - t0
        error_count = sum(
            1
            for _, item in ((row.get("model_outputs") or {}).items())
            if str((item or {}).get("error") or "").strip()
        )
        print(
            f"[{_ts()}] [split:done] sample_id={sample_id} elapsed={elapsed:.2f}s errors={error_count}",
            flush=True,
        )
        return row

    if not parallel:
        print(
            f"[{_ts()}] [split] mode=serial rows={len(clean_rows)} model={model} timeout=45 retries=1",
            flush=True,
        )
        llm = LLMClient(provider="openrouter", model=model, timeout_seconds=45, max_retries=1)
        pipe = PrepareClaimsPipeline(
            source_path=out_dir / "unused.source.jsonl",
            clean_path=out_dir / "unused.clean.jsonl",
            split_path=out_dir / "unused.split.jsonl",
            group_path=out_dir / "unused.group.jsonl",
            llm_client=llm,
        )
        return pipe.run_split_stage(clean_rows)

    print(
        f"[{_ts()}] [split] mode=parallel rows={len(clean_rows)} workers={workers} model={model} timeout=45 retries=1",
        flush=True,
    )
    out: list[dict[str, Any] | None] = [None] * len(clean_rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_idx = {ex.submit(run_one, row): i for i, row in enumerate(clean_rows)}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                out[idx] = fut.result()
            except Exception as e:
                row = clean_rows[idx]
                sample = row.get("sample") or {}
                sample_id = str(sample.get("id") or "").strip()
                print(f"[{_ts()}] [split:error] idx={idx} sample_id={sample_id} error={e}", flush=True)
                fallback_outputs: dict[str, dict[str, Any]] = {}
                for seq_key, item in (row.get("model_outputs") or {}).items():
                    fallback_outputs[str(seq_key)] = {
                        "model_sequence_id": int((item or {}).get("model_sequence_id") or 0),
                        "model_name": str((item or {}).get("model_name") or ""),
                        "atomic_claims": [],
                        "error": f"split_runner_error: {e}",
                    }
                out[idx] = {"sample": row.get("sample"), "model_outputs": fallback_outputs}
    return [x for x in out if x is not None]


def main() -> None:
    cfg = _load_config()
    print(
        f"[{_ts()}] [entry:start] target={cfg.run_target_stage} reuse_from={cfg.reuse_from_stage} "
        f"input={cfg.input_path} run_name={cfg.run_name or '(auto)'}",
        flush=True,
    )
    validate_stage_control(cfg)
    _validate_entry_stage_policy(cfg)
    validate_paths(cfg)
    ensure_ddgs_dependency()

    if not cfg.run_name.strip():
        cfg = SimpleQARunConfig(**{**cfg.__dict__, "run_name": _safe_name(cfg.input_path.stem)})

    out_dir = resolve_output_dir(cfg)
    source_rows = _build_source_rows(cfg)
    print(f"[{_ts()}] [entry:input] selected_rows={len(source_rows)} out_dir={out_dir}", flush=True)
    source_data_path = out_dir / "source_data.selected.jsonl"
    clean_path = out_dir / "cleaned_answer.jsonl"
    split_path = out_dir / "split_claims.jsonl"
    write_jsonl(source_data_path, source_rows)

    target_rank = STAGE_RANK[cfg.run_target_stage]
    reuse_rank = STAGE_RANK[cfg.reuse_from_stage]

    if reuse_rank >= 0:
        clean_path = cfg.existing_clean_path or clean_path
    if reuse_rank >= 1:
        split_path = cfg.existing_split_path or split_path

    if target_rank >= 0 and reuse_rank < 0:
        source_records = build_source_records(source_rows)
        records = source_records
        if RETRY_ONLY_ERRORS and clean_path.exists():
            existing_clean_rows = read_jsonl(clean_path)
            records = _project_clean_retry_records(source_records, existing_clean_rows)
            print(
                f"[{_ts()}] [entry:clean] retry_only_errors enabled existing_rows={len(existing_clean_rows)} retry_records={len(records)}",
                flush=True,
            )
            if not records:
                clean_rows = existing_clean_rows
                print(
                    f"[{_ts()}] [entry:clean] skip no failed items path={clean_path} errors={_count_stage_errors(clean_rows)}",
                    flush=True,
                )
            else:
                clean_rows = []
        else:
            clean_rows = []
        print(f"[{_ts()}] [entry:clean] start records={len(records)}", flush=True)
        t0 = time.perf_counter()
        if records:
            rerun_clean_rows = _run_clean_records(
                records,
                model=cfg.clean_model,
                workers=cfg.max_prepare_workers,
                out_dir=out_dir,
                parallel=cfg.enable_prepare_parallel,
            )
            if RETRY_ONLY_ERRORS and clean_path.exists():
                existing_clean_rows = read_jsonl(clean_path)
                clean_rows = _merge_stage_rows(existing_clean_rows, rerun_clean_rows)
            else:
                clean_rows = rerun_clean_rows
            write_jsonl(clean_path, clean_rows)
        print(
            f"[{_ts()}] [entry:clean] done rows={len(clean_rows)} elapsed={time.perf_counter() - t0:.2f}s "
            f"errors={_count_stage_errors(clean_rows)} path={clean_path}",
            flush=True,
        )

    if target_rank >= 1 and reuse_rank < 1:
        clean_rows_for_split = read_jsonl(clean_path)
        split_input_rows = clean_rows_for_split
        if RETRY_ONLY_ERRORS and split_path.exists():
            existing_split_rows = read_jsonl(split_path)
            split_input_rows = _project_split_retry_rows(clean_rows_for_split, existing_split_rows)
            print(
                f"[{_ts()}] [entry:split] retry_only_errors enabled existing_rows={len(existing_split_rows)} retry_rows={len(split_input_rows)}",
                flush=True,
            )
            if not split_input_rows:
                split_rows = existing_split_rows
                print(
                    f"[{_ts()}] [entry:split] skip no failed items path={split_path} errors={_count_stage_errors(split_rows)}",
                    flush=True,
                )
            else:
                split_rows = []
        else:
            split_rows = []
        print(f"[{_ts()}] [entry:split] start rows={len(split_input_rows)}", flush=True)
        t0 = time.perf_counter()
        if split_input_rows:
            rerun_split_rows = _run_split_rows(
                split_input_rows,
                model=cfg.split_model,
                workers=cfg.max_prepare_workers,
                out_dir=out_dir,
                parallel=cfg.enable_prepare_parallel,
            )
            if RETRY_ONLY_ERRORS and split_path.exists():
                existing_split_rows = read_jsonl(split_path)
                split_rows = _merge_stage_rows(existing_split_rows, rerun_split_rows)
            else:
                split_rows = rerun_split_rows
            write_jsonl(split_path, split_rows)
        print(
            f"[{_ts()}] [entry:split] done rows={len(split_rows)} elapsed={time.perf_counter() - t0:.2f}s "
            f"errors={_count_stage_errors(split_rows)} path={split_path}",
            flush=True,
        )

    topic_grounding_path = cfg.existing_topic_grounding_path
    topic_guidance_path = cfg.existing_topic_guidance_path

    if target_rank >= 3:
        if topic_grounding_path is None or not topic_grounding_path.exists():
            raise FileNotFoundError("retrieve stage requires existing topic_grounding.jsonl (prefetch prepared in advance)")
        if topic_guidance_path is None or not topic_guidance_path.exists():
            raise FileNotFoundError("retrieve stage requires existing topic_guidance.jsonl (prefetch prepared in advance)")

    if target_rank >= 3 and reuse_rank < 3:
        run_retrieval_from_existing_files(
            split_path=split_path,
            clean_path=clean_path,
            out_dir=out_dir,
            retrieve_model=cfg.retrieve_model,
            max_retrieval_round=cfg.max_retrieval_round,
            max_recoverable_retries_per_round=cfg.max_recoverable_retries_per_round,
            max_must_answer_retries=cfg.max_must_answer_retries,
            max_snippet_per_search=cfg.max_snippet_per_search,
            max_initial_snippets=cfg.max_initial_snippets,
            enable_bm25_initial_select=cfg.enable_bm25_initial_select,
            duplicate_threshold=cfg.duplicate_threshold,
            no_result_defer_threshold=cfg.no_result_defer_threshold,
            no_progress_defer_threshold=cfg.no_progress_defer_threshold,
            top_k_freq_snippets=cfg.top_k_freq_snippets,
            query_cache_jaccard_threshold=cfg.query_cache_jaccard_threshold,
            enable_topic_parallel=cfg.enable_topic_parallel,
            max_topic_workers=cfg.max_topic_workers,
            topic_grounding_path=topic_grounding_path,
            topic_guidance_path=topic_guidance_path,
        )

    print(f"[done] target={cfg.run_target_stage} reuse_from={cfg.reuse_from_stage} out_dir={out_dir}")


if __name__ == "__main__":
    main()
