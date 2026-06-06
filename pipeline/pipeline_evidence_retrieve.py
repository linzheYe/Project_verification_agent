#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from scripts.data_io import parse_json_object_robust, read_jsonl, write_jsonl
from scripts.ddg_search import ensure_ddgs_dependency, search_snippets_with_ddg
from scripts.evidence_registry import (
    ClaimSnippetStatus,
    QuestionEvidencePoolRegistry,
    find_best_pool_duplicate_by_content,
    filter_non_duplicate_snippets_by_content,
    rank_snippets_by_bm25,
)
from scripts.llm_api import LLMClient, LLMConnectivityError
from scripts.prompt_template import (
    FACTUAL_LABEL,
    NEI_LABEL,
    NON_FACTUAL_LABEL,
    build_must_have_answer_prompt,
    build_next_search_or_answer_prompt,
    build_snippet_filter_prompt,
)


class FatalPipelineError(RuntimeError):
    """Fatal connectivity error that should stop the whole run immediately."""


class EvidenceRetrievePipeline:
    """Evidence retrieval pipeline for grouped atomic claims.

    Design intent:
    - Keep one readable control flow for iterative retrieval.
    - Split only stable boundaries: prompt build, LLM call, DDG search, registry state.
    """

    def __init__(
        self,
        *,
        split_path: Path,
        clean_path: Path,
        retrieval_trace_path: Path,
        claim_status_path: Path,
        question_pool_path: Path,
        evidence_results_path: Path,
        high_freq_snippets_path: Path | None = None,
        timing_summary_path: Path | None = None,
        llm_client: LLMClient | None = None,
        max_retrieval_round: int = 3,
        max_snippet_per_search: int = 8,
        max_initial_snippets: int = 8,
        enable_bm25_initial_select: bool = True,
        enable_bm25_query_stopwords: bool = False,
        duplicate_threshold: float = 0.9,
        max_recoverable_retries_per_round: int = 3,
        max_must_answer_retries: int = 2,
        no_result_defer_threshold: int = 1,
        no_progress_defer_threshold: int = 2,
        top_k_freq_snippets: int = 5,
        query_cache_jaccard_threshold: float = 0.8,
        enable_fixed_topic_grounding_evidence: bool = False,
        enable_filter_dataset_snippets: bool = False,
        topic_grounding_path: Path | None = None,
        topic_guidance_path: Path | None = None,
    ) -> None:
        self.split_path = split_path
        self.clean_path = clean_path
        self.retrieval_trace_path = retrieval_trace_path
        self.claim_status_path = claim_status_path
        self.question_pool_path = question_pool_path
        self.evidence_results_path = evidence_results_path
        self.high_freq_snippets_path = high_freq_snippets_path
        self.timing_summary_path = timing_summary_path
        self.llm = llm_client or LLMClient()

        self.max_retrieval_round = max(1, int(max_retrieval_round))
        self.max_snippet_per_search = max(1, int(max_snippet_per_search))
        self.max_initial_snippets = max(1, int(max_initial_snippets))
        self.enable_bm25_initial_select = bool(enable_bm25_initial_select)
        self.enable_bm25_query_stopwords = bool(enable_bm25_query_stopwords)
        self.duplicate_threshold = float(duplicate_threshold)
        self.max_recoverable_retries_per_round = max(0, int(max_recoverable_retries_per_round))
        self.max_must_answer_retries = max(0, int(max_must_answer_retries))
        self.no_result_defer_threshold = max(1, int(no_result_defer_threshold))
        self.no_progress_defer_threshold = max(1, int(no_progress_defer_threshold))
        self.top_k_freq_snippets = max(1, int(top_k_freq_snippets))
        self.query_cache_jaccard_threshold = max(0.0, min(1.0, float(query_cache_jaccard_threshold)))
        self.enable_fixed_topic_grounding_evidence = bool(enable_fixed_topic_grounding_evidence)
        self.enable_filter_dataset_snippets = bool(enable_filter_dataset_snippets)
        self.topic_grounding_path = topic_grounding_path
        self.topic_guidance_path = topic_guidance_path

        self.question_pool_registry = QuestionEvidencePoolRegistry()
        self._query_cache_by_question: dict[str, dict[str, dict[str, Any]]] = {}
        self._timing: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._prefetch_evidence_by_query: dict[str, list[dict[str, Any]]] = {}
        self._topic_guidance_by_query: dict[str, str] = {}
        self._topic_guidance_by_question: dict[str, str] = {}
        self._seeded_question_keys: set[str] = set()

    def _add_time(self, key: str, seconds: float) -> None:
        """Accumulate elapsed time for one stage key."""
        self._timing[key] = self._timing.get(key, 0.0) + max(0.0, float(seconds))

    def _inc_count(self, key: str, n: int = 1) -> None:
        """Accumulate call counts for one stage key."""
        self._counts[key] = self._counts.get(key, 0) + int(n)

    @staticmethod
    def _truncate_jsonl(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    @staticmethod
    def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    @staticmethod
    def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
        sample = row.get("sample") or {}
        return str(sample.get("id") or ""), str(sample.get("query") or "")

    @staticmethod
    def _iter_model_outputs(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return sorted(
            (row.get("model_outputs") or {}).items(),
            key=lambda kv: int((kv[1] or {}).get("model_sequence_id") or 0),
        )

    @staticmethod
    def _safe_int(x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _build_background_map(clean_rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], str]:
        """Map (sample_id, query, model_sequence_id) -> cleaned_model_answer."""
        mp: dict[tuple[str, str, int], str] = {}
        for row in clean_rows:
            sample = row.get("sample") or {}
            sample_id = str(sample.get("id") or "")
            query = str(sample.get("query") or "")
            for _, model_output in sorted(
                (row.get("model_outputs") or {}).items(),
                key=lambda kv: int((kv[1] or {}).get("model_sequence_id") or 0),
            ):
                model_sequence_id = int(model_output.get("model_sequence_id") or 0)
                background_text = str(model_output.get("cleaned_model_answer") or "").strip()
                mp[(sample_id, query, model_sequence_id)] = background_text
        return mp

    @staticmethod
    def _extract_claim_pairs_from_atomic_claims(model_output: dict[str, Any]) -> list[tuple[int, str]]:
        """Extract ordered (atomic_claim_sequence_number, claim_text) from split-stage atomic_claims."""
        claims = model_output.get("atomic_claims") or []
        pairs: list[tuple[int, str]] = []
        for idx, claim_obj in enumerate(claims, start=1):
            if not isinstance(claim_obj, dict):
                continue
            claim_text = str(claim_obj.get("atomic_claim") or "").strip()
            if not claim_text:
                continue
            seq = EvidenceRetrievePipeline._safe_int(claim_obj.get("atomic_claim_sequence_number"), idx)
            pairs.append((seq, claim_text))
        pairs.sort(key=lambda x: x[0])
        return pairs

    @staticmethod
    def _snippet_brief(snippet: dict[str, Any]) -> dict[str, Any]:
        out = {
            "snippet_id": str(snippet.get("snippet_id") or "").strip(),
            "url": str(snippet.get("url") or "").strip(),
            "title": str(snippet.get("title") or "").strip(),
            "content": str(snippet.get("content") or "").strip(),
        }
        if "is_fixed_topic_evidence" in snippet:
            out["is_fixed_topic_evidence"] = bool(snippet.get("is_fixed_topic_evidence"))
        if "present_added_round" in snippet:
            try:
                out["present_added_round"] = int(snippet.get("present_added_round") or 0)
            except Exception:
                out["present_added_round"] = 0
        return out

    @staticmethod
    def _snippet_brief_without_round(snippet: dict[str, Any]) -> dict[str, Any]:
        return {
            "snippet_id": str(snippet.get("snippet_id") or "").strip(),
            "url": str(snippet.get("url") or "").strip(),
            "title": str(snippet.get("title") or "").strip(),
            "content": str(snippet.get("content") or "").strip(),
        }

    @staticmethod
    def _normalize_query_for_similarity(query: str) -> str:
        return " ".join(re.findall(r"[\w\u4e00-\u9fff]+", str(query or "").lower()))

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        sa = set(a.split())
        sb = set(b.split())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _get_query_cache(self, question_key: str) -> dict[str, dict[str, Any]]:
        return self._query_cache_by_question.setdefault(question_key, {})

    def _cache_query_outcome(
        self,
        *,
        question_key: str,
        search_query: str,
        status: str,
        direct_snippets: list[dict[str, Any]] | None = None,
    ) -> None:
        norm = self._normalize_query_for_similarity(search_query)
        if not norm:
            return
        brief_direct_snippets: list[dict[str, Any]] = []
        for x in (direct_snippets or []):
            brief = self._snippet_brief(x)
            brief.pop("present_added_round", None)
            brief_direct_snippets.append(brief)
        self._get_query_cache(question_key)[norm] = {
            "norm_query": norm,
            "raw_query": search_query,
            "status": status,  # hit_with_direct | hit_but_empty | pending
            "direct_snippets": brief_direct_snippets,
        }

    def _find_similar_cached_query(
        self,
        *,
        question_key: str,
        search_query: str,
    ) -> dict[str, Any] | None:
        norm = self._normalize_query_for_similarity(search_query)
        if not norm:
            return None
        best: dict[str, Any] | None = None
        best_score = -1.0
        for item in self._get_query_cache(question_key).values():
            score = self._jaccard_similarity(norm, str(item.get("norm_query") or ""))
            if score > best_score:
                best = item
                best_score = score
        if best is None or best_score <= self.query_cache_jaccard_threshold:
            return None
        out = dict(best)
        out["similarity"] = best_score
        return out

    @staticmethod
    def _build_search_history_for_next_action(status: ClaimSnippetStatus) -> list[dict[str, Any]]:
        present_by_id: dict[str, dict[str, Any]] = {}
        for sn in (status.selected_present_snippets or []):
            sid = str((sn or {}).get("snippet_id") or "").strip()
            if not sid or not re.fullmatch(r"S\d+", sid):
                continue
            present_by_id[sid] = {
                "snippet_id": sid,
                "url": str((sn or {}).get("url") or "").strip(),
                "title": str((sn or {}).get("title") or "").strip(),
                "content": str((sn or {}).get("content") or "").strip(),
            }

        history: list[dict[str, Any]] = []
        for row in (status.round_logs or []):
            search_query = str((row or {}).get("search_query") or "").strip()
            if not search_query:
                continue
            valid_ids: list[str] = []
            valid_snippets: list[dict[str, Any]] = []
            for raw_sid in ((row or {}).get("direct_relevant_snippet_ids") or []):
                sid = str(raw_sid or "").strip()
                if not sid or not re.fullmatch(r"S\d+", sid):
                    continue
                snippet = present_by_id.get(sid)
                if snippet is None:
                    continue
                valid_ids.append(sid)
                valid_snippets.append(dict(snippet))
            history.append(
                {
                    "round_index": int((row or {}).get("round_index") or 0),
                    "search_query": search_query,
                    "newly_added_present_snippet_ids": valid_ids,
                    "newly_added_present_snippets": valid_snippets,
                }
            )
        return history

    @staticmethod
    def _merge_present_snippets(
        current_snippets: list[dict[str, Any]],
        new_snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge snippet lists by snippet_id while preserving prior order."""
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for sn in current_snippets + new_snippets:
            sid = str(sn.get("snippet_id") or "").strip()
            if not sid or sid in seen:
                continue
            merged.append(dict(sn))
            seen.add(sid)
        return merged

    @staticmethod
    def _with_present_round(
        snippets: list[dict[str, Any]],
        round_index: int,
    ) -> list[dict[str, Any]]:
        tagged: list[dict[str, Any]] = []
        round_value = max(0, int(round_index))
        for sn in snippets:
            row = dict(sn)
            row["present_added_round"] = round_value
            tagged.append(row)
        return tagged

    @staticmethod
    def _split_fixed_topic_snippets(
        snippets: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        fixed: list[dict[str, Any]] = []
        regular: list[dict[str, Any]] = []
        for sn in snippets:
            if bool((sn or {}).get("is_fixed_topic_evidence")):
                fixed.append(dict(sn))
            else:
                regular.append(dict(sn))
        return fixed, regular

    def _select_present_snippets_for_claim(
        self,
        *,
        question_key: str,
        claim_text: str,
    ) -> list[dict[str, Any]]:
        pool = self.question_pool_registry.get_question_pool_snippets(question_key)
        if not self.enable_fixed_topic_grounding_evidence:
            # Legacy behavior: topic grounding snippets stay inside the same pool
            # and compete with other snippets for the initial selection budget.
            if not pool:
                return []
            if not self.enable_bm25_initial_select:
                return [self._snippet_brief(sn) for sn in pool]
            return [self._snippet_brief(sn) for sn in rank_snippets_by_bm25(
                claim_text,
                pool,
                top_k=max(0, int(self.max_initial_snippets)),
                remove_query_stopwords=self.enable_bm25_query_stopwords,
            )]

        fixed_pool_snippets, regular_pool_snippets = self._split_fixed_topic_snippets(pool)

        if not regular_pool_snippets:
            regular_selected: list[dict[str, Any]] = []
        elif not self.enable_bm25_initial_select:
            regular_selected = [dict(x) for x in regular_pool_snippets]
        else:
            regular_selected = rank_snippets_by_bm25(
                claim_text,
                regular_pool_snippets,
                top_k=max(0, int(self.max_initial_snippets)),
                remove_query_stopwords=self.enable_bm25_query_stopwords,
            )

        return self._merge_present_snippets(
            [self._snippet_brief(sn) for sn in fixed_pool_snippets],
            [self._snippet_brief(sn) for sn in regular_selected],
        )

    def _add_fixed_topic_evidence(
        self,
        *,
        question_key: str,
        present_snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.enable_fixed_topic_grounding_evidence:
            return [dict(x) for x in present_snippets]
        pool = self.question_pool_registry.get_question_pool_snippets(question_key)
        fixed_pool_snippets, _ = self._split_fixed_topic_snippets(pool)
        return self._merge_present_snippets(
            [self._snippet_brief(sn) for sn in fixed_pool_snippets],
            [self._snippet_brief(sn) for sn in present_snippets],
        )

    @staticmethod
    def _is_dataset_snippet(snippet: dict[str, Any]) -> bool:
        title = str((snippet or {}).get("title") or "").lower()
        url = str((snippet or {}).get("url") or "").lower()
        return "dataset" in title or "dataset" in url

    def _filter_dataset_snippets(
        self,
        snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.enable_filter_dataset_snippets:
            return [dict(x) for x in snippets]
        # Optional hygiene filter for DDG web results only; prefetch topic grounding is untouched.
        return [dict(x) for x in snippets if not self._is_dataset_snippet(x)]

    def _load_prefetch_resources(self) -> None:
        """Load optional prefetch evidence and topic guidance files once per run."""
        self._prefetch_evidence_by_query = {}
        self._topic_guidance_by_query = {}
        self._topic_guidance_by_question = {}

        if self.topic_grounding_path is not None and self.topic_grounding_path.exists():
            for row in read_jsonl(self.topic_grounding_path):
                query = str(row.get("query") or "").strip()
                if not query:
                    continue
                evidence_items = row.get("evidence_items") or []
                snippets: list[dict[str, Any]] = []
                for ev in evidence_items:
                    if not isinstance(ev, dict):
                        continue
                    content = str(ev.get("content") or ev.get("relevant_text") or "").strip()
                    if not content:
                        continue
                    snippet_id = str(ev.get("snippet_id") or "").strip().upper()
                    if not re.fullmatch(r"S\d+", snippet_id):
                        raise ValueError(
                            "prefetch evidence item missing valid snippet_id (expected S* format): "
                            f"query={query}, snippet_id={snippet_id!r}"
                        )
                    snippets.append(
                        {
                            "snippet_id": snippet_id,
                            "url": str(ev.get("source_url") or ev.get("url") or "").strip(),
                            "title": str(ev.get("source_title") or ev.get("title") or "").strip(),
                            "content": content,
                            "is_fixed_topic_evidence": True,
                        }
                    )
                if snippets:
                    self._prefetch_evidence_by_query[query] = snippets
                brief = str(row.get("topic_brief") or "").strip()
                if brief:
                    self._topic_guidance_by_query[query] = brief

        if self.topic_guidance_path is not None and self.topic_guidance_path.exists():
            for row in read_jsonl(self.topic_guidance_path):
                query = str(row.get("query") or "").strip()
                guidance = str(row.get("topic_guidance") or "").strip()
                if query and guidance:
                    self._topic_guidance_by_query[query] = guidance

    def _seed_prefetch_evidence_into_question_pool(
        self,
        *,
        question_key: str,
        query: str,
    ) -> None:
        """Inject topic-level prefetch evidence as initial pool snippets once per question key."""
        if question_key in self._seeded_question_keys:
            return
        seed_snippets = self._prefetch_evidence_by_query.get(query) or []
        if seed_snippets:
            self.question_pool_registry.add_snippets_to_question_pool(question_key, seed_snippets)
        guidance_raw = self._topic_guidance_by_query.get(query, "")
        if re.search(r"\bT\d+_E\d+\b", guidance_raw):
            raise ValueError(
                "topic_guidance contains legacy T*_E* ids. "
                "New strict format requires S* ids only."
            )
        self._topic_guidance_by_question[question_key] = guidance_raw
        self._seeded_question_keys.add(question_key)

    def _call_llm_json_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        stage: str = "generic",
    ) -> tuple[dict[str, Any], str, str]:
        raw = self.llm.call(system_prompt, user_prompt, temperature=temperature)
        parsed, parse_mode = parse_json_object_robust(raw)
        self._inc_count(f"parse_mode_{stage}_{parse_mode}")
        return parsed, parse_mode, raw

    def _call_llm_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        stage: str = "generic",
    ) -> dict[str, Any]:
        parsed, _, _ = self._call_llm_json_with_meta(
            system_prompt,
            user_prompt,
            temperature=temperature,
            stage=stage,
        )
        return parsed

    @staticmethod
    def _is_fatal_connectivity_error(error: Exception) -> bool:
        if isinstance(error, LLMConnectivityError):
            return True
        msg = str(error).lower()
        fatal_signals = [
            "unable to connect to proxy",
            "proxyerror",
            "failed to establish a new connection",
            "connection refused",
            "max retries exceeded",
            "openrouter_request_error",
        ]
        return any(x in msg for x in fatal_signals)

    def _raise_if_fatal_error(self, error: Exception, *, stage: str) -> None:
        if self._is_fatal_connectivity_error(error):
            raise FatalPipelineError(f"{stage}_fatal_connectivity_error: {error}") from error

    @staticmethod
    def _is_recoverable_ddg_error(error: Exception) -> bool:
        msg = str(error).lower()
        return "no results found" in msg

    def _finalize_with_must_answer(
        self,
        *,
        status: ClaimSnippetStatus,
        claim_text: str,
        background_text: str,
        present_snippets: list[dict[str, Any]],
    ) -> None:
        """Finalize one claim via MUST_HAVE_ANSWER using provided snippet context."""
        t0 = time.perf_counter()
        must_sys, must_user_base = build_must_have_answer_prompt(
            {
                "claim": claim_text,
                "disambiguation_text": background_text,
                "topic_guidance": self._topic_guidance_by_question.get(
                    status.query,
                    self._topic_guidance_by_query.get(status.query, ""),
                ),
                "present_snippets_with_ids": [
                    self._snippet_brief_without_round(x) for x in present_snippets
                ],
            }
        )

        last_error: Exception | None = None
        last_raw = ""
        total_attempts = self.max_must_answer_retries + 1
        for attempt in range(total_attempts):
            must_user = must_user_base
            if attempt > 0:
                feedback = {
                    "had_error": True,
                    "error_type": "must_answer_parse_or_schema_error",
                    "error_message": str(last_error or ""),
                    "retry_attempt": attempt,
                    "instruction": (
                        "The previous attempt failed due to parse/schema issues. "
                        "Read last_step_feedback and retry now. "
                        "Return only valid JSON object with keys: "
                        "final_answer, reasoning, evidence_snippet_ids."
                    ),
                    "format_instruction": (
                        "Return only valid JSON object with keys: "
                        "final_answer, reasoning, evidence_snippet_ids."
                    ),
                    "previous_raw_output": (last_raw or "")[:2000],
                }
                must_user = must_user_base + "\n\nlast_step_feedback:\n" + json.dumps(
                    feedback,
                    ensure_ascii=False,
                    indent=2,
                )
            try:
                must_obj, _, raw = self._call_llm_json_with_meta(
                    must_sys,
                    must_user,
                    temperature=0.1,
                    stage="must_answer",
                )
                last_raw = raw
                must_payload = self._parse_must_answer(must_obj)

                present_id_set = {str(x.get("snippet_id") or "") for x in present_snippets}
                evidence_ids = [sid for sid in (must_payload.get("evidence_snippet_ids") or []) if sid in present_id_set]

                status.finalize_answer(
                    final_answer=must_payload["final_answer"],
                    final_reason=must_payload.get("reason") or "",
                    evidence_snippet_ids=evidence_ids,
                    finished_by_max_round=True,
                )
                self._add_time("llm_must_answer_seconds", time.perf_counter() - t0)
                self._inc_count("llm_must_answer_calls")
                if attempt > 0:
                    self._inc_count("must_answer_retry_success")
                return
            except Exception as e:
                self._raise_if_fatal_error(e, stage="must_answer")
                last_error = e
                if attempt < self.max_must_answer_retries:
                    self._inc_count("must_answer_retry_attempts")
                    continue
                self._inc_count("must_answer_retry_exhausted")
                status.set_error((status.error + " | " if status.error else "") + f"must_answer_error: {e}")
                status.finalize_answer(
                    final_answer=NEI_LABEL,
                    final_reason="insufficient or invalid structured answer from MUST_HAVE_ANSWER",
                    evidence_snippet_ids=[],
                    finished_by_max_round=True,
                )
                return

    def _finalize_deferred_error_claim(self, status: ClaimSnippetStatus) -> ClaimSnippetStatus:
        """Finalize a previously deferred claim using latest question pool."""
        question_key = status.query
        latest_pool = self.question_pool_registry.get_question_pool_snippets(question_key)
        bm25_seed = self._select_present_snippets_for_claim(
            question_key=question_key,
            claim_text=status.claim_text,
        )
        present_snippets = self._merge_present_snippets(
            [self._snippet_brief(sn) for sn in status.selected_present_snippets],
            [self._snippet_brief(sn) for sn in bm25_seed],
        )
        if not present_snippets:
            present_snippets = self._add_fixed_topic_evidence(
                question_key=question_key,
                present_snippets=[self._snippet_brief(sn) for sn in latest_pool],
            )
        else:
            present_snippets = self._add_fixed_topic_evidence(
                question_key=question_key,
                present_snippets=present_snippets,
            )
        status.update_present_snippets(present_snippets)

        self._finalize_with_must_answer(
            status=status,
            claim_text=status.claim_text,
            background_text=status.background_text,
            present_snippets=present_snippets,
        )
        self._inc_count("claims_deferred_then_must_answer")
        self._inc_count("claims_processed")
        return status

    def _extract_rationale_snippet_ids(self, rationale: str) -> list[str]:
        ids = re.findall(r"\bS\d+\b", str(rationale or ""), flags=re.IGNORECASE)
        out: list[str] = []
        seen: set[str] = set()
        for sid in ids:
            sid_up = sid.upper()
            if sid_up in seen:
                continue
            seen.add(sid_up)
            out.append(sid_up)
        return out

    def _build_high_freq_stats(self, statuses: list[ClaimSnippetStatus]) -> list[dict[str, Any]]:
        by_question_counts: dict[str, dict[str, int]] = {}
        sample_ids_by_question: dict[str, list[str]] = {}
        for st in statuses:
            if st.claim_status != "done":
                continue
            qk = st.query
            sid = str(st.sample_id or "").strip()
            if sid:
                sample_ids_by_question.setdefault(qk, [])
                if sid not in sample_ids_by_question[qk]:
                    sample_ids_by_question[qk].append(sid)
            rationale_ids = self._extract_rationale_snippet_ids(st.final_reason)
            if not rationale_ids:
                continue
            mp = by_question_counts.setdefault(qk, {})
            for sid in rationale_ids:
                mp[sid] = int(mp.get(sid, 0)) + 1

        rows: list[dict[str, Any]] = []
        for qk, sid_counts in by_question_counts.items():
            pool = self.question_pool_registry.get_question_pool_snippets(qk)
            pool_by_id = {str(x.get("snippet_id") or "").strip().upper(): x for x in pool}
            query = qk
            q_sample_ids = sample_ids_by_question.get(qk, [])
            for sid, support_count in sid_counts.items():
                sn = pool_by_id.get(sid, {})
                score = int(support_count or 0)
                rows.append(
                    {
                        "sample_id": q_sample_ids[0] if q_sample_ids else "",
                        "sample_ids": list(q_sample_ids),
                        "query": query,
                        "snippet_id": sid,
                        "url": str(sn.get("url") or ""),
                        "title": str(sn.get("title") or ""),
                        "support_count": score,
                        "score": score,
                    }
                )
        rows.sort(
            key=lambda x: (
                x.get("query") or "",
                -int(x.get("score") or 0),
                x.get("snippet_id") or "",
            )
        )
        return rows

    def _parse_next_action(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Parse NEXT_SEARCH_OR_ANSWER action output with explicit validation."""
        action = str(obj.get("action") or "").strip().lower()
        if not action and str(obj.get("final_answer") or "").strip():
            action = "answer"
        if action == "search":
            search_query = str(obj.get("search_query") or "").strip()
            missing_fact_to_verify = str(obj.get("missing_fact_to_verify") or "").strip()
            # Backward compatibility: old prompt/schema used `reason`.
            if not missing_fact_to_verify:
                missing_fact_to_verify = str(obj.get("reason") or "").strip()
            if not missing_fact_to_verify:
                missing_fact_to_verify = str(obj.get("reasoning") or "").strip()
            if not search_query:
                raise ValueError("next_action_missing_search_query")
            return {
                "action": "search",
                "search_query": search_query,
                "missing_fact_to_verify": missing_fact_to_verify,
            }

        if action == "answer":
            final_answer = str(obj.get("final_answer") or "").strip()
            reason = str(obj.get("reason") or "").strip()
            if not reason:
                reason = str(obj.get("reasoning") or "").strip()
            evidence_snippet_ids = obj.get("evidence_snippet_ids") or []
            if final_answer not in {FACTUAL_LABEL, NON_FACTUAL_LABEL}:
                raise ValueError("next_action_invalid_final_answer")
            return {
                "action": "answer",
                "final_answer": final_answer,
                "reason": reason,
                "evidence_snippet_ids": [str(x).strip() for x in evidence_snippet_ids if str(x).strip()],
            }

        raise ValueError("next_action_invalid_action")

    def _parse_must_answer(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Parse MUST_HAVE_ANSWER output into a normalized answer payload."""
        # Compatible with both schemas:
        # - {"final_answer":..., "reason":..., "evidence_snippet_ids":[...]}
        # - {"final_answer":..., "reasoning":..., "evidence_snippet_ids":[...]}
        # - {"action":"answer", ...}
        action = str(obj.get("action") or "").strip().lower()
        if action and action != "answer":
            raise ValueError("must_answer_action_must_be_answer")

        final_answer = str(obj.get("final_answer") or "").strip()
        reason = str(obj.get("reason") or "").strip()
        if not reason:
            reason = str(obj.get("reasoning") or "").strip()
        evidence_snippet_ids = obj.get("evidence_snippet_ids") or []
        if final_answer not in {FACTUAL_LABEL, NON_FACTUAL_LABEL, NEI_LABEL}:
            raise ValueError("must_answer_invalid_final_answer")

        return {
            "final_answer": final_answer,
            "reason": reason,
            "evidence_snippet_ids": [str(x).strip() for x in evidence_snippet_ids if str(x).strip()],
        }

    @staticmethod
    def _assign_candidate_ids(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with_ids: list[dict[str, Any]] = []
        for idx, sn in enumerate(snippets, start=1):
            content = str((sn or {}).get("content") or "").strip()
            if not content:
                continue
            with_ids.append(
                {
                    "candidate_id": f"C{idx}",
                    "url": str((sn or {}).get("url") or "").strip(),
                    "title": str((sn or {}).get("title") or "").strip(),
                    "content": content,
                }
            )
        return with_ids

    def _run_single_claim(
        self,
        *,
        sample_id: str,
        query: str,
        model_sequence_id: int,
        model_name: str,
        model_answer: str,
        question_group_name: str,
        claim_sequence_number: int,
        claim_text: str,
        background_text: str,
    ) -> ClaimSnippetStatus:
        claim_start = time.perf_counter()
        defer_must_for_error = False
        defer_on_no_evidence = False
        consecutive_no_result_rounds = 0
        consecutive_no_progress_rounds = 0
        retry_attempts_for_this_round = 0
        last_error_type = ""
        last_error_message = ""
        failed_query = ""
        question_key = self.question_pool_registry.get_or_create_question_pool(
            sample_id=sample_id,
            query=query,
            question_group_name=question_group_name,
        )
        self._seed_prefetch_evidence_into_question_pool(question_key=question_key, query=query)

        status = ClaimSnippetStatus(
            sample_id=sample_id,
            query=query,
            model_sequence_id=model_sequence_id,
            model_name=model_name,
            model_answer=model_answer,
            atomic_claim_sequence_number=claim_sequence_number,
            question_group_name=question_group_name,
            claim_text=claim_text,
            background_text=background_text,
        )

        t0 = time.perf_counter()
        initial_snippets = self._select_present_snippets_for_claim(
            question_key=question_key,
            claim_text=claim_text,
        )
        if self.enable_bm25_initial_select:
            self._add_time("bm25_initial_select_seconds", time.perf_counter() - t0)
            self._inc_count("bm25_initial_select_calls")
        present_snippets = self._with_present_round(
            [self._snippet_brief(sn) for sn in initial_snippets],
            0,
        )
        status.initialize_retrieval(present_snippets)

        # Iterative retrieval loop with early stop.
        while status.retrieval_round < self.max_retrieval_round:
            try:
                t0 = time.perf_counter()
                do_not_repeat_queries = list(status.searched_queries)
                if failed_query and failed_query not in do_not_repeat_queries:
                    do_not_repeat_queries.append(failed_query)
                next_sys, next_user = build_next_search_or_answer_prompt(
                    {
                        "claim": claim_text,
                        "disambiguation_text": background_text,
                        "topic_guidance": self._topic_guidance_by_question.get(
                            question_key,
                            self._topic_guidance_by_query.get(query, ""),
                        ),
                        "search_history": self._build_search_history_for_next_action(status),
                        "last_error_type": last_error_type,
                        "last_error_message": last_error_message,
                        "failed_query": failed_query,
                        "retry_attempts_for_this_round": retry_attempts_for_this_round,
                        "do_not_repeat_queries": do_not_repeat_queries,
                    }
                )
                next_obj = self._call_llm_json(next_sys, next_user, temperature=0.2, stage="next_action")
                action_payload = self._parse_next_action(next_obj)
                self._add_time("llm_next_action_seconds", time.perf_counter() - t0)
                self._inc_count("llm_next_action_calls")
            except Exception as e:
                self._raise_if_fatal_error(e, stage="next_action")
                status.set_error(f"next_action_error: {e}")
                defer_must_for_error = True
                break

            if action_payload["action"] == "answer":
                evidence_ids = action_payload.get("evidence_snippet_ids") or []
                present_id_set = {str(x.get("snippet_id") or "") for x in present_snippets}
                evidence_ids = [sid for sid in evidence_ids if sid in present_id_set]

                status.finalize_answer(
                    final_answer=action_payload["final_answer"],
                    final_reason=action_payload.get("reason") or "",
                    evidence_snippet_ids=evidence_ids,
                    finished_by_max_round=False,
                )
                self._add_time("single_claim_total_seconds", time.perf_counter() - claim_start)
                self._inc_count("claims_processed")
                return status

            search_query = action_payload["search_query"]
            if search_query in status.searched_queries:
                if retry_attempts_for_this_round < self.max_recoverable_retries_per_round:
                    retry_attempts_for_this_round += 1
                    last_error_type = "duplicate_search_query_from_next_action"
                    last_error_message = "next action repeated a previously used search query"
                    failed_query = search_query
                    self._inc_count("recoverable_retry_attempts")
                    continue
                status.set_error(
                    "duplicate_search_query_from_next_action"
                    f" | recoverable_retry_exhausted={self.max_recoverable_retries_per_round}"
                )
                break

            cached = self._find_similar_cached_query(question_key=question_key, search_query=search_query)
            if cached is not None:
                self._inc_count("query_cache_similarity_hits")
                cache_status = str(cached.get("status") or "")
                if cache_status == "hit_with_direct":
                    cached_direct = [self._snippet_brief(x) for x in (cached.get("direct_snippets") or [])]
                    cached_direct = self._with_present_round(
                        cached_direct,
                        status.retrieval_round + 1,
                    )
                    present_snippets = self._merge_present_snippets(present_snippets, cached_direct)
                    present_snippets = self._add_fixed_topic_evidence(
                        question_key=question_key,
                        present_snippets=present_snippets,
                    )
                    status.update_present_snippets(present_snippets)
                    status.record_search_round(
                        search_query=search_query,
                        missing_fact_to_verify=str(action_payload.get("missing_fact_to_verify") or ""),
                        selected_present_snippet_ids=[str(x.get("snippet_id") or "") for x in present_snippets],
                        newly_added_pool_snippet_ids=[],
                        direct_relevant_snippet_ids=[str(x.get("snippet_id") or "") for x in cached_direct],
                        round_source="query_cache_reuse",
                        round_outcome="reuse_direct",
                        extra_trace_fields={
                            "reused_from_query": str(cached.get("raw_query") or ""),
                            "reused_from_cache_status": cache_status,
                            "reused_similarity": round(float(cached.get("similarity") or 0.0), 4),
                            "reused_from_direct_snippet_ids": [
                                str(x.get("snippet_id") or "") for x in cached_direct
                            ],
                        },
                    )
                    self._inc_count("query_cache_reuse_ddg_skipped")
                    continue
                if cache_status in {"pending", "hit_but_empty"}:
                    status.record_search_round(
                        search_query=search_query,
                        missing_fact_to_verify=str(action_payload.get("missing_fact_to_verify") or ""),
                        selected_present_snippet_ids=[str(x.get("snippet_id") or "") for x in present_snippets],
                        newly_added_pool_snippet_ids=[],
                        direct_relevant_snippet_ids=[],
                        round_source="query_cache_reuse",
                        round_outcome="reuse_pending_propagated",
                        extra_trace_fields={
                            "reused_from_query": str(cached.get("raw_query") or ""),
                            "reused_from_cache_status": cache_status,
                            "reused_similarity": round(float(cached.get("similarity") or 0.0), 4),
                        },
                    )
                    status.set_error("deferred_no_evidence: similar_query_cache_pending_or_empty")
                    defer_on_no_evidence = True
                    self._inc_count("query_cache_pending_propagated")
                    break

            try:
                t0 = time.perf_counter()
                raw_snippets = search_snippets_with_ddg(
                    search_query,
                    max_results=self.max_snippet_per_search,
                )
                raw_snippets = self._filter_dataset_snippets(raw_snippets)
                self._add_time("ddg_search_seconds", time.perf_counter() - t0)
                self._inc_count("ddg_search_calls")
            except Exception as e:
                self._raise_if_fatal_error(e, stage="ddg_search")
                if (
                    self._is_recoverable_ddg_error(e)
                    and retry_attempts_for_this_round < self.max_recoverable_retries_per_round
                ):
                    retry_attempts_for_this_round += 1
                    last_error_type = "ddg_search_error"
                    last_error_message = str(e)
                    failed_query = search_query
                    self._inc_count("recoverable_retry_attempts")
                    continue
                status.set_error(f"ddg_search_error: {e}")
                break

            if not raw_snippets:
                consecutive_no_result_rounds += 1
                consecutive_no_progress_rounds += 1
                self._inc_count("no_results_events")
                self._cache_query_outcome(
                    question_key=question_key,
                    search_query=search_query,
                    status="hit_but_empty",
                    direct_snippets=[],
                )
                status.record_search_round(
                    search_query=search_query,
                    missing_fact_to_verify=str(action_payload.get("missing_fact_to_verify") or ""),
                    selected_present_snippet_ids=[str(x.get("snippet_id") or "") for x in present_snippets],
                    newly_added_pool_snippet_ids=[],
                    direct_relevant_snippet_ids=[],
                    round_source="ddg_search",
                    round_outcome="ddg_no_results",
                )
                if consecutive_no_result_rounds >= self.no_result_defer_threshold:
                    defer_on_no_evidence = True
                    status.set_error("deferred_no_evidence: reached no_result_defer_threshold")
                    self._cache_query_outcome(
                        question_key=question_key,
                        search_query=search_query,
                        status="pending",
                        direct_snippets=[],
                    )
                    break
                continue
            consecutive_no_result_rounds = 0

            # Let the LLM see the full post-filter search results first. Pool dedup
            # happens only after the model has decided which snippets should enter the pool.
            candidate_snippets = self._assign_candidate_ids(raw_snippets)

            kept_pool_candidate_ids: list[str] = []
            kept_direct_candidate_ids: list[str] = []
            if candidate_snippets:
                try:
                    t0 = time.perf_counter()
                    filter_sys, filter_user = build_snippet_filter_prompt(
                        {
                            "claim": claim_text,
                            "topic": question_group_name,
                            "disambiguation_text": background_text,
                            "current_round_snippets": candidate_snippets,
                            "existing_claim_snippets": [
                                self._snippet_brief_without_round(x) for x in present_snippets
                            ],
                        }
                    )
                    filter_obj = self._call_llm_json(filter_sys, filter_user, temperature=0.2, stage="snippet_filter")
                    kept_pool_candidate_ids = [
                        str(x).strip()
                        for x in (filter_obj.get("evidence_pool_candidate_ids") or [])
                        if str(x).strip()
                    ]
                    kept_direct_candidate_ids = [
                        str(x).strip()
                        for x in (filter_obj.get("direct_relevant_candidate_ids") or [])
                        if str(x).strip()
                    ]
                    self._add_time("llm_snippet_filter_seconds", time.perf_counter() - t0)
                    self._inc_count("llm_snippet_filter_calls")
                except Exception as e:
                    self._raise_if_fatal_error(e, stage="snippet_filter")
                    status.set_error(f"snippet_filter_error: {e}")
                    status.record_search_round(
                        search_query=search_query,
                        missing_fact_to_verify=str(action_payload.get("missing_fact_to_verify") or ""),
                        selected_present_snippet_ids=[str(x.get("snippet_id") or "") for x in present_snippets],
                        newly_added_pool_snippet_ids=[],
                        direct_relevant_snippet_ids=[],
                        round_source="ddg_search",
                        round_outcome="snippet_filter_error",
                    )
                    defer_must_for_error = True
                    break
            if defer_must_for_error:
                break

            candidate_by_id = {str(x["candidate_id"]): x for x in candidate_snippets}
            pool_candidates_pre_dedup = [candidate_by_id[cid] for cid in kept_pool_candidate_ids if cid in candidate_by_id]
            direct_candidates = [candidate_by_id[cid] for cid in kept_direct_candidate_ids if cid in candidate_by_id]

            pool_before = self.question_pool_registry.get_question_pool_snippets(question_key)
            t0 = time.perf_counter()
            pool_candidates = filter_non_duplicate_snippets_by_content(
                pool_candidates_pre_dedup,
                pool_before,
                threshold=self.duplicate_threshold,
            )
            self._add_time("snippet_dedup_seconds", time.perf_counter() - t0)
            self._inc_count("snippet_dedup_calls")

            added_snippets = self.question_pool_registry.add_snippets_to_question_pool(
                question_key, pool_candidates
            )
            added_map_by_candidate_id: dict[str, dict[str, Any]] = {}
            for cand, added in zip(pool_candidates, added_snippets):
                added_map_by_candidate_id[str(cand.get("candidate_id") or "")] = dict(added)

            new_direct_snippets_with_ids: list[dict[str, Any]] = []
            for cand in direct_candidates:
                cid = str(cand.get("candidate_id") or "")
                if cid in added_map_by_candidate_id:
                    new_direct_snippets_with_ids.append(self._snippet_brief(added_map_by_candidate_id[cid]))
                    continue
                best_existing = find_best_pool_duplicate_by_content(
                    cand,
                    pool_before,
                    threshold=self.duplicate_threshold,
                )
                if best_existing is not None:
                    new_direct_snippets_with_ids.append(self._snippet_brief(best_existing))

            new_direct_snippets_with_ids = self._with_present_round(
                new_direct_snippets_with_ids,
                status.retrieval_round + 1,
            )
            present_snippets = self._merge_present_snippets(present_snippets, new_direct_snippets_with_ids)
            present_snippets = self._add_fixed_topic_evidence(
                question_key=question_key,
                present_snippets=present_snippets,
            )
            status.update_present_snippets(present_snippets)
            status.record_search_round(
                search_query=search_query,
                missing_fact_to_verify=str(action_payload.get("missing_fact_to_verify") or ""),
                selected_present_snippet_ids=[str(x.get("snippet_id") or "") for x in present_snippets],
                newly_added_pool_snippet_ids=[str(x.get("snippet_id") or "") for x in added_snippets],
                direct_relevant_snippet_ids=[str(x.get("snippet_id") or "") for x in new_direct_snippets_with_ids],
                round_source="ddg_search",
                round_outcome="ddg_with_progress" if (added_snippets or new_direct_snippets_with_ids) else "ddg_no_progress",
            )
            has_progress = bool(added_snippets) or bool(new_direct_snippets_with_ids)
            self._cache_query_outcome(
                question_key=question_key,
                search_query=search_query,
                status="hit_with_direct" if new_direct_snippets_with_ids else "hit_but_empty",
                direct_snippets=new_direct_snippets_with_ids,
            )
            if has_progress:
                consecutive_no_progress_rounds = 0
            else:
                consecutive_no_progress_rounds += 1
                self._inc_count("no_progress_events")
                if consecutive_no_progress_rounds >= self.no_progress_defer_threshold:
                    defer_on_no_evidence = True
                    status.set_error("deferred_no_evidence: reached no_progress_defer_threshold")
                    self._cache_query_outcome(
                        question_key=question_key,
                        search_query=search_query,
                        status="pending",
                        direct_snippets=[],
                    )
                    break
            retry_attempts_for_this_round = 0
            last_error_type = ""
            last_error_message = ""
            failed_query = ""

        if defer_must_for_error:
            # Defer MUST fallback to a second pass after all non-error claims finish.
            status.claim_status = "deferred_error"
            self._inc_count("claims_deferred_on_error")
            self._add_time("single_claim_total_seconds", time.perf_counter() - claim_start)
            return status
        if defer_on_no_evidence:
            status.claim_status = "deferred_no_evidence"
            self._inc_count("claims_deferred_no_evidence")
            self._add_time("single_claim_total_seconds", time.perf_counter() - claim_start)
            return status

        # Force final answer when max retrieval rounds are exhausted.
        self._finalize_with_must_answer(
            status=status,
            claim_text=claim_text,
            background_text=background_text,
            present_snippets=present_snippets,
        )

        self._add_time("single_claim_total_seconds", time.perf_counter() - claim_start)
        self._inc_count("claims_processed")
        return status

    def _build_rows_from_status(self, status: ClaimSnippetStatus) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        claim_status_row = status.to_json_record()

        retrieval_trace_rows: list[dict[str, Any]] = []
        for round_log in status.round_logs:
            clean_round_log = {k: v for k, v in round_log.items() if k != "missing_fact_to_verify"}
            retrieval_trace_rows.append(
                {
                    "sample_id": status.sample_id,
                    "query": status.query,
                    "model_sequence_id": status.model_sequence_id,
                    "atomic_claim_sequence_number": status.atomic_claim_sequence_number,
                    "claim_text": status.claim_text,
                    "claim_retrieval_round": int(round_log.get("round_index") or 0),
                    **clean_round_log,
                }
            )

        present_by_id = {
            str(sn.get("snippet_id") or ""): sn
            for sn in status.selected_present_snippets
            if str(sn.get("snippet_id") or "").strip()
        }
        cited_snippet_ids: list[str] = []
        seen_cited_ids: set[str] = set()
        for sid in list(status.evidence_snippet_ids) + self._extract_rationale_snippet_ids(status.final_reason):
            sid_norm = str(sid or "").strip().upper()
            if not sid_norm or sid_norm in seen_cited_ids:
                continue
            seen_cited_ids.add(sid_norm)
            cited_snippet_ids.append(sid_norm)

        evidence_snippets = [
            {
                "snippet_id": sid,
                "url": str(present_by_id[sid].get("url") or ""),
                "title": str(present_by_id[sid].get("title") or ""),
                "content": str(present_by_id[sid].get("content") or ""),
            }
            for sid in cited_snippet_ids
            if sid in present_by_id
        ]
        evidence_result_row = {
            "sample_id": status.sample_id,
            "query": status.query,
            "model_sequence_id": status.model_sequence_id,
            "model": status.model_name,
            "model_answer": status.model_answer,
            "atomic_claim_sequence_number": status.atomic_claim_sequence_number,
            "atomic_claim": status.claim_text,
            "searched_queries": list(status.searched_queries),
            "retrieval_round": status.retrieval_round,
            "used_must_have_answer": status.finished_by_max_round,
            "auto_verdict": status.final_answer,
            "verdict_rationale": status.final_reason,
            "rationale_mentioned_snippet_ids": self._extract_rationale_snippet_ids(status.final_reason),
            "evidence_snippets": evidence_snippets,
            "round_sources": [str(x.get("round_source") or "") for x in status.round_logs],
            "has_query_cache_reuse": any(str(x.get("round_source") or "") == "query_cache_reuse" for x in status.round_logs),
            "had_ddg_search_pending": "deferred_no_evidence" in str(status.error or ""),
            "error": status.error,
        }
        return claim_status_row, retrieval_trace_rows, evidence_result_row

    def run(self) -> None:
        pipeline_start = time.perf_counter()
        # Fail early if DDG dependency is missing.
        ensure_ddgs_dependency()

        split_rows = read_jsonl(self.split_path)
        clean_rows = read_jsonl(self.clean_path)
        self._load_prefetch_resources()
        background_map = self._build_background_map(clean_rows)

        # Streaming mode: truncate output files first, then append per-claim.
        self._truncate_jsonl(self.retrieval_trace_path)
        self._truncate_jsonl(self.claim_status_path)
        self._truncate_jsonl(self.evidence_results_path)
        self._truncate_jsonl(self.question_pool_path)

        all_statuses: list[ClaimSnippetStatus] = []
        deferred_error_statuses: list[ClaimSnippetStatus] = []
        top_freq_ids_by_question: dict[str, list[str]] = {}

        for row in split_rows:
            sample = row.get("sample") or {}
            sample_id = str(sample.get("id") or "").strip()
            query = str(sample.get("query") or "").strip()
            if not sample_id or not query:
                continue

            for _, model_output in self._iter_model_outputs(row):
                model_sequence_id = int(model_output.get("model_sequence_id") or 0)
                model_name = str(model_output.get("model_name") or "").strip()
                model_answer = str(model_output.get("model_answer") or "").strip()
                background_text = background_map.get((sample_id, query, model_sequence_id), "")
                if not background_text:
                    background_text = query

                question_group_name = query
                for claim_seq_num, claim_text in self._extract_claim_pairs_from_atomic_claims(model_output):
                    status = self._run_single_claim(
                        sample_id=sample_id,
                        query=query,
                        model_sequence_id=model_sequence_id,
                        model_name=model_name,
                        model_answer=model_answer,
                        question_group_name=question_group_name,
                        claim_sequence_number=claim_seq_num,
                        claim_text=claim_text,
                        background_text=background_text,
                    )
                    all_statuses.append(status)
                    if status.claim_status != "done":
                        deferred_error_statuses.append(status)
                    else:
                        claim_status_row, trace_rows, evidence_row = self._build_rows_from_status(status)
                        self._append_jsonl_row(self.claim_status_path, claim_status_row)
                        for tr in trace_rows:
                            self._append_jsonl_row(self.retrieval_trace_path, tr)
                        self._append_jsonl_row(self.evidence_results_path, evidence_row)
                        write_jsonl(self.question_pool_path, self.question_pool_registry.dump_question_pool_rows())

        # Build high-frequency snippet ids from completed claims before second pass.
        freq_rows_pre = self._build_high_freq_stats(all_statuses)
        for row in freq_rows_pre:
            qk = str(row.get("query") or "")
            sid = str(row.get("snippet_id") or "").strip().upper()
            if not qk or not sid:
                continue
            top_freq_ids_by_question.setdefault(qk, [])
            if sid not in top_freq_ids_by_question[qk]:
                top_freq_ids_by_question[qk].append(sid)

        # Second pass: finalize deferred claims after all non-error claims are done.
        for status in deferred_error_statuses:
            qk = status.query
            latest_pool = self.question_pool_registry.get_question_pool_snippets(qk)
            pool_by_id = {str(x.get("snippet_id") or "").strip().upper(): x for x in latest_pool}
            top_freq_ids_used: list[str] = []
            if not self.enable_fixed_topic_grounding_evidence:
                # Legacy behavior: second pass prioritizes high-frequency snippets,
                # then falls back to the latest full topic evidence pool.
                top_freq = [
                    pool_by_id[sid]
                    for sid in top_freq_ids_by_question.get(qk, [])[: self.top_k_freq_snippets]
                    if sid in pool_by_id
                ]
                top_freq_ids_used = [str(sn.get("snippet_id") or "") for sn in top_freq]
                if top_freq:
                    present = self._merge_present_snippets(
                        [self._snippet_brief(sn) for sn in top_freq],
                        [self._snippet_brief(sn) for sn in latest_pool],
                    )
                    status.update_present_snippets(present)
            else:
                top_freq = [
                    pool_by_id[sid]
                    for sid in top_freq_ids_by_question.get(qk, [])[: self.top_k_freq_snippets]
                    if sid in pool_by_id and not bool((pool_by_id[sid] or {}).get("is_fixed_topic_evidence"))
                ]
                top_freq_ids_used = [str(sn.get("snippet_id") or "") for sn in top_freq]
                if top_freq or status.selected_present_snippets:
                    # Fixed topic grounding is always retained as background context.
                    # The top-k competition is only among non-fixed retrieval evidence.
                    existing_non_fixed = [
                        self._snippet_brief(sn)
                        for sn in status.selected_present_snippets
                        if not bool((sn or {}).get("is_fixed_topic_evidence"))
                    ]
                    present = self._merge_present_snippets(
                        existing_non_fixed,
                        [self._snippet_brief(sn) for sn in top_freq],
                    )
                    present = self._add_fixed_topic_evidence(
                        question_key=qk,
                        present_snippets=present,
                    )
                    status.update_present_snippets(present)
            self._finalize_deferred_error_claim(status)
            status.record_search_round(
                search_query="",
                missing_fact_to_verify="",
                selected_present_snippet_ids=[
                    str(x.get("snippet_id") or "") for x in (status.selected_present_snippets or [])
                ],
                newly_added_pool_snippet_ids=[],
                direct_relevant_snippet_ids=[str(x) for x in (status.evidence_snippet_ids or [])],
                round_source="deferred_finalize",
                round_outcome="must_answer_with_top_freq" if top_freq_ids_used else "must_answer_after_deferred",
                extra_trace_fields={
                    "top_freq_snippet_ids_used": list(top_freq_ids_used),
                    "final_answer": str(status.final_answer or ""),
                    "final_evidence_snippet_ids": [str(x) for x in (status.evidence_snippet_ids or [])],
                    "resolved_from_error": str(status.error or ""),
                },
            )
            claim_status_row, trace_rows, evidence_row = self._build_rows_from_status(status)
            self._append_jsonl_row(self.claim_status_path, claim_status_row)
            for tr in trace_rows:
                self._append_jsonl_row(self.retrieval_trace_path, tr)
            self._append_jsonl_row(self.evidence_results_path, evidence_row)
            write_jsonl(self.question_pool_path, self.question_pool_registry.dump_question_pool_rows())

        for status in all_statuses:
            if not status.finished_by_max_round:
                self._inc_count("claims_early_stop")
            else:
                self._inc_count("claims_forced_must_answer")

        if self.high_freq_snippets_path is not None:
            freq_rows_final = self._build_high_freq_stats(all_statuses)
            write_jsonl(self.high_freq_snippets_path, freq_rows_final)

        self._add_time("pipeline_total_seconds", time.perf_counter() - pipeline_start)

        if self.timing_summary_path is not None:
            summary = {
                "timing_seconds": dict(sorted(self._timing.items(), key=lambda x: x[0])),
                "counts": dict(sorted(self._counts.items(), key=lambda x: x[0])),
            }
            self.timing_summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.timing_summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(f"done: split={self.split_path}")
        print(f"- retrieval_trace -> {self.retrieval_trace_path}")
        print(f"- claim_status -> {self.claim_status_path}")
        print(f"- question_pool -> {self.question_pool_path}")
        print(f"- evidence_results -> {self.evidence_results_path}")
        if self.high_freq_snippets_path is not None:
            print(f"- high_freq_snippets -> {self.high_freq_snippets_path}")
        if self.timing_summary_path is not None:
            print(f"- timing_summary -> {self.timing_summary_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    run_dir = base_dir / "data_prepare_outputs" / "runs" / "round2"

    pipeline = EvidenceRetrievePipeline(
        split_path=run_dir / "split_claims.second_try.jsonl",
        clean_path=run_dir / "cleaned_answer.second_try.jsonl",
        retrieval_trace_path=run_dir / "retrieval_trace.second_try.jsonl",
        claim_status_path=run_dir / "claim_snippets_status.second_try.jsonl",
        question_pool_path=run_dir / "question_evidence_pool.second_try.jsonl",
        evidence_results_path=run_dir / "evidence_results.second_try.jsonl",
        timing_summary_path=run_dir / "timing_summary.second_try.json",
        llm_client=LLMClient(provider="openrouter"),
        max_retrieval_round=3,
        max_recoverable_retries_per_round=3,
        max_must_answer_retries=2,
        max_snippet_per_search=8,
        max_initial_snippets=8,
        enable_bm25_initial_select=True,
        duplicate_threshold=0.9,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
