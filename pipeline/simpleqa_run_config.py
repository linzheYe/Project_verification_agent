from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE_RANK = {"none": -1, "clean": 0, "split": 1, "prefetch": 2, "retrieve": 3, "rubrics": 4}


@dataclass(frozen=True)
class SimpleQARunConfig:
    # Input / output
    input_mode: str  # model_jsonl | rubrics_jsonl
    input_path: Path
    output_base_dir: Path
    run_name: str

    # Stage control
    run_target_stage: str  # clean | split | prefetch | retrieve | rubrics
    reuse_from_stage: str  # none | clean | split | prefetch | retrieve

    # Reuse paths
    existing_clean_path: Path | None
    existing_split_path: Path | None
    existing_topic_grounding_path: Path | None
    existing_topic_guidance_path: Path | None
    existing_evidence_results_path: Path | None

    # Models
    clean_model: str
    split_model: str
    prefetch_model: str
    retrieve_model: str

    # Prepare parallel
    enable_prepare_parallel: bool
    max_prepare_workers: int

    # Prefetch params
    prefetch_enable_topic_parallel: bool
    prefetch_max_topic_workers: int
    prefetch_wiki_query_count: int
    prefetch_web_query_count: int
    prefetch_snippets_per_query: int
    prefetch_max_selected_candidates: int
    prefetch_fetch_timeout_seconds: int
    prefetch_max_page_text_chars: int
    prefetch_near_dup_threshold: float

    # Retrieve params
    max_retrieval_round: int
    max_recoverable_retries_per_round: int
    max_must_answer_retries: int
    max_snippet_per_search: int
    max_initial_snippets: int
    enable_bm25_initial_select: bool
    duplicate_threshold: float
    no_result_defer_threshold: int
    no_progress_defer_threshold: int
    top_k_freq_snippets: int
    query_cache_jaccard_threshold: float
    enable_topic_parallel: bool
    max_topic_workers: int


def validate_stage_control(cfg: SimpleQARunConfig) -> None:
    target = cfg.run_target_stage.strip().lower()
    reuse = cfg.reuse_from_stage.strip().lower()
    if target not in STAGE_RANK or target == "none":
        raise ValueError(f"invalid run_target_stage: {cfg.run_target_stage}")
    if reuse not in STAGE_RANK:
        raise ValueError(f"invalid reuse_from_stage: {cfg.reuse_from_stage}")
    if STAGE_RANK[reuse] > STAGE_RANK[target]:
        raise ValueError("reuse_from_stage cannot be later than run_target_stage")


def validate_paths(cfg: SimpleQARunConfig) -> None:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"input file not found: {cfg.input_path}")
    if cfg.run_name.strip() and ("/" in cfg.run_name or "\\" in cfg.run_name):
        raise ValueError("run_name must be a directory name, not a path")

    reuse_rank = STAGE_RANK[cfg.reuse_from_stage]
    if reuse_rank >= 0 and (cfg.existing_clean_path is None or not cfg.existing_clean_path.exists()):
        raise FileNotFoundError("reuse_from_stage>=clean but existing_clean_path is missing")
    if reuse_rank >= 1 and (cfg.existing_split_path is None or not cfg.existing_split_path.exists()):
        raise FileNotFoundError("reuse_from_stage>=split but existing_split_path is missing")
    if reuse_rank >= 2:
        if cfg.existing_topic_grounding_path is None or not cfg.existing_topic_grounding_path.exists():
            raise FileNotFoundError("reuse_from_stage>=prefetch but existing_topic_grounding_path is missing")
        if cfg.existing_topic_guidance_path is None or not cfg.existing_topic_guidance_path.exists():
            raise FileNotFoundError("reuse_from_stage>=prefetch but existing_topic_guidance_path is missing")
    if reuse_rank >= 3 and (
        cfg.existing_evidence_results_path is None or not cfg.existing_evidence_results_path.exists()
    ):
        raise FileNotFoundError("reuse_from_stage>=retrieve but existing_evidence_results_path is missing")


def resolve_output_dir(cfg: SimpleQARunConfig) -> Path:
    out_dir = cfg.output_base_dir / cfg.run_name.strip() if cfg.run_name.strip() else cfg.output_base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
