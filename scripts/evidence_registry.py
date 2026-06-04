from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


QuestionKey = str


@dataclass
class QuestionPoolMeta:
    """Metadata for one question-level evidence pool."""

    query: str
    sample_ids: list[str] = field(default_factory=list)


@dataclass
class ClaimSnippetStatus:
    """State tracker for one claim across multi-round retrieval.

    Why a class:
    - Retrieval loop needs stable per-claim state for debugging and replay.
    - Keeping all fields together avoids scattered temporary variables.
    """

    sample_id: str
    query: str
    model_sequence_id: int
    model_name: str
    model_answer: str
    atomic_claim_sequence_number: int
    question_group_name: str
    claim_text: str
    background_text: str

    claim_status: str = "pending"
    retrieval_round: int = 0
    searched_queries: list[str] = field(default_factory=list)
    selected_initial_snippets: list[dict[str, Any]] = field(default_factory=list)
    selected_present_snippets: list[dict[str, Any]] = field(default_factory=list)
    round_logs: list[dict[str, Any]] = field(default_factory=list)

    final_answer: str = ""
    final_reason: str = ""
    evidence_snippet_ids: list[str] = field(default_factory=list)
    finished_by_max_round: bool = False
    error: str = ""

    def initialize_retrieval(self, selected_initial_snippets: list[dict[str, Any]]) -> None:
        """Initialize per-claim retrieval state before first decision.

        selected_initial_snippets are chosen from topic evidence pool by BM25.
        """
        self.claim_status = "retrieving"
        self.selected_initial_snippets = [dict(x) for x in selected_initial_snippets]
        self.selected_present_snippets = [dict(x) for x in selected_initial_snippets]

    def record_search_round(
        self,
        *,
        search_query: str,
        missing_fact_to_verify: str,
        selected_present_snippet_ids: list[str],
        newly_added_pool_snippet_ids: list[str],
        direct_relevant_snippet_ids: list[str],
        round_source: str = "ddg_search",
        round_outcome: str = "",
    ) -> None:
        """Record one search iteration after DDG + filter steps complete."""
        self.retrieval_round += 1
        self.searched_queries.append(search_query)
        self.round_logs.append(
            {
                "round_index": self.retrieval_round,
                "search_query": search_query,
                "missing_fact_to_verify": str(missing_fact_to_verify or "").strip(),
                "selected_present_snippet_ids": list(selected_present_snippet_ids),
                "newly_added_pool_snippet_ids": list(newly_added_pool_snippet_ids),
                "direct_relevant_snippet_ids": list(direct_relevant_snippet_ids),
                "round_source": str(round_source or "").strip(),
                "round_outcome": str(round_outcome or "").strip(),
            }
        )

    def update_present_snippets(self, present_snippets: list[dict[str, Any]]) -> None:
        """Update the snippet set used for the next decision round."""
        self.selected_present_snippets = [dict(x) for x in present_snippets]

    def finalize_answer(
        self,
        *,
        final_answer: str,
        final_reason: str,
        evidence_snippet_ids: list[str],
        finished_by_max_round: bool,
    ) -> None:
        """Close the claim with final answer and supporting snippet ids."""
        self.claim_status = "done"
        self.final_answer = str(final_answer or "").strip()
        self.final_reason = str(final_reason or "").strip()
        self.evidence_snippet_ids = [str(x) for x in evidence_snippet_ids if str(x).strip()]
        self.finished_by_max_round = bool(finished_by_max_round)

    def set_error(self, error_text: str) -> None:
        """Store explicit error message for traceability."""
        self.error = str(error_text or "").strip()

    def to_json_record(self) -> dict[str, Any]:
        """Serialize status into one JSONL row."""
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "model_sequence_id": self.model_sequence_id,
            "model_name": self.model_name,
            "model_answer": self.model_answer,
            "atomic_claim_sequence_number": self.atomic_claim_sequence_number,
            "claim_text": self.claim_text,
            "background_text": self.background_text,
            "claim_status": self.claim_status,
            "retrieval_round": self.retrieval_round,
            "searched_queries": list(self.searched_queries),
            "selected_initial_snippets": [dict(x) for x in self.selected_initial_snippets],
            "selected_present_snippets": [dict(x) for x in self.selected_present_snippets],
            "round_logs": [dict(x) for x in self.round_logs],
            "final_answer": self.final_answer,
            "final_reason": self.final_reason,
            "evidence_snippet_ids": list(self.evidence_snippet_ids),
            "finished_by_max_round": self.finished_by_max_round,
            "error": self.error,
        }


class QuestionEvidencePoolRegistry:
    """Registry for all question-level evidence pools.

    This class is intentionally narrow:
    - It stores snippets per question.
    - It provides BM25-based initial snippet selection for new claim processing.
    - It does not decide search/query/action policies.
    """

    def __init__(self) -> None:
        self._pool_by_question: dict[QuestionKey, list[dict[str, Any]]] = {}
        self._meta_by_question: dict[QuestionKey, QuestionPoolMeta] = {}
        self._next_snippet_id_by_question: dict[QuestionKey, int] = {}

    def get_or_create_question_pool(
        self,
        *,
        sample_id: str,
        query: str,
        question_group_name: str,
    ) -> QuestionKey:
        """Create pool entry on first use and return stable question key.

        Key policy:
        - Pool is keyed only by query so snippets can be reused across sample_ids
          under the same question/topic.
        """
        key: QuestionKey = str(query)
        if key not in self._pool_by_question:
            self._pool_by_question[key] = []
            self._next_snippet_id_by_question[key] = 1
        if key not in self._meta_by_question:
            self._meta_by_question[key] = QuestionPoolMeta(
                query=str(query),
            )
        sid = str(sample_id or "").strip()
        if sid:
            ids = self._meta_by_question[key].sample_ids
            if sid not in ids:
                ids.append(sid)
        return key

    def get_question_pool_snippets(self, question_key: QuestionKey) -> list[dict[str, Any]]:
        """Return current question evidence pool snippets."""
        return [dict(x) for x in self._pool_by_question.get(question_key, [])]

    def add_snippets_to_question_pool(
        self,
        question_key: QuestionKey,
        snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append new snippets into pool and assign/preserve stable snippet_id.

        Note:
        - Input snippets should already be deduplicated against current pool.
        - If input snippet already provides snippet_id like S12, preserve it.
        - Otherwise, snippet_id is assigned in insertion order for reproducible tracing.
        """
        pool = self._pool_by_question.setdefault(question_key, [])
        self._next_snippet_id_by_question.setdefault(question_key, 1)
        existing_ids = {str(x.get("snippet_id") or "").strip().upper() for x in pool}

        added: list[dict[str, Any]] = []
        for sn in snippets:
            content = str((sn or {}).get("content") or "").strip()
            if not content:
                continue
            incoming_id = str((sn or {}).get("snippet_id") or "").strip()
            incoming_upper = incoming_id.upper()
            if incoming_upper and incoming_upper in existing_ids:
                continue

            snippet_id = ""
            if re.fullmatch(r"S\d+", incoming_upper):
                snippet_id = incoming_upper
                num = int(incoming_upper[1:])
                self._next_snippet_id_by_question[question_key] = max(
                    self._next_snippet_id_by_question[question_key],
                    num + 1,
                )
            else:
                snippet_id = f"S{self._next_snippet_id_by_question[question_key]}"
                self._next_snippet_id_by_question[question_key] += 1

            row = {
                "snippet_id": snippet_id,
                "url": str((sn or {}).get("url") or "").strip(),
                "title": str((sn or {}).get("title") or "").strip(),
                "content": content,
            }
            if "is_fixed_topic_evidence" in (sn or {}):
                row["is_fixed_topic_evidence"] = bool((sn or {}).get("is_fixed_topic_evidence"))
            pool.append(row)
            added.append(dict(row))
            existing_ids.add(snippet_id.upper())
        return added

    def select_initial_snippets_for_claim(
        self,
        question_key: QuestionKey,
        claim_text: str,
        max_initial_snippets: int = 5,
        use_bm25: bool = True,
    ) -> list[dict[str, Any]]:
        """Select initial snippets from old topic pool before processing a claim.

        Ranking policy:
        - If use_bm25=True, return BM25 top-k snippets.
        - If use_bm25=False, return all snippets in topic pool.
        """
        pool = self._pool_by_question.get(question_key, [])
        if not pool:
            return []
        if not use_bm25:
            return [dict(x) for x in pool]
        return rank_snippets_by_bm25(
            claim_text=claim_text,
            snippets=pool,
            top_k=max(0, int(max_initial_snippets)),
        )

    def dump_question_pool_rows(self) -> list[dict[str, Any]]:
        """Export all in-memory question evidence pools into JSONL row objects.

        When this is called:
        - Called once near the end of pipeline run, before writing
          `question_evidence_pool.jsonl`.

        What it returns:
        - One row per query key.
        - Each row contains `pool_snippets` with stable `snippet_id`, `url`,
          `title`, and `content`.

        Why this is useful:
        - The question pool is mutable in memory during retrieval loops.
        - This export step snapshots the final state for reproducibility and debugging.
        - Without this function, downstream analysis cannot see what evidence was
          accumulated per question after the run.
        """
        rows: list[dict[str, Any]] = []
        for question_key, snippets in self._pool_by_question.items():
            meta = self._meta_by_question.get(question_key)
            if not meta:
                continue
            rows.append(
                {
                    "sample_id": meta.sample_ids[0] if meta.sample_ids else "",
                    "sample_ids": list(meta.sample_ids),
                    "query": meta.query,
                    "pool_snippets": [dict(x) for x in snippets],
                }
            )
        return rows


def detect_text_language_simple(text: str) -> str:
    """Simple Chinese/English detector for tokenizer choice.

    Rule of thumb:
    - If CJK characters are present, use Chinese path.
    - Otherwise use English path.
    """
    t = str(text or "")
    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    return "en"


def tokenize_for_bm25(text: str, lang: str) -> list[str]:
    """Tokenize text for BM25 with lightweight zh/en handling."""
    s = str(text or "").lower()
    if not s:
        return []

    if lang == "zh":
        zh_chars = re.findall(r"[\u4e00-\u9fff]", s)
        zh_bigrams = ["".join(zh_chars[i : i + 2]) for i in range(max(0, len(zh_chars) - 1))]
        en_words = re.findall(r"[a-z0-9]+", s)
        return zh_chars + zh_bigrams + en_words

    return re.findall(r"[a-z0-9]+", s)


def rank_snippets_by_bm25(
    claim_text: str,
    snippets: list[dict[str, Any]],
    top_k: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Rank snippets by BM25 and return top_k rows.

    Why keep this in evidence_registry.py:
    - This ranking is only used by claim initialization against topic pool.
    - Putting it here keeps retrieval flow easy to follow in one place.
    """
    if top_k <= 0:
        return []

    docs: list[list[str]] = []
    for sn in snippets:
        content = str((sn or {}).get("content") or "").strip()
        lang = detect_text_language_simple(content)
        docs.append(tokenize_for_bm25(content, lang))

    query_lang = detect_text_language_simple(claim_text)
    query_terms = tokenize_for_bm25(claim_text, query_lang)
    if not query_terms:
        return []

    n_docs = len(docs)
    if n_docs == 0:
        return []

    avgdl = sum(len(d) for d in docs) / n_docs if n_docs > 0 else 0.0
    avgdl = max(avgdl, 1.0)

    # Document frequency table.
    df: dict[str, int] = {}
    for doc_tokens in docs:
        for term in set(doc_tokens):
            df[term] = df.get(term, 0) + 1

    scored: list[tuple[float, int]] = []
    for idx, doc_tokens in enumerate(docs):
        doc_len = max(len(doc_tokens), 1)
        tf: dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for term in query_terms:
            term_df = df.get(term, 0)
            if term_df <= 0:
                continue
            # BM25 IDF (Robertson-Sparck Jones style).
            idf = math.log((n_docs - term_df + 0.5) / (term_df + 0.5) + 1.0)
            term_tf = tf.get(term, 0)
            if term_tf <= 0:
                continue
            denom = term_tf + k1 * (1.0 - b + b * doc_len / avgdl)
            score += idf * (term_tf * (k1 + 1.0) / max(denom, 1e-9))

        scored.append((score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rows: list[dict[str, Any]] = []
    for _, idx in scored[:top_k]:
        top_rows.append(dict(snippets[idx]))
    return top_rows


def filter_non_duplicate_snippets_by_content(
    new_snippets: list[dict[str, Any]],
    topic_pool_snippets: list[dict[str, Any]],
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """Keep only snippets whose content is not near-duplicate to pool.

    Duplicate rule:
    - Compare only `content` as requested.
    - Jaccard similarity > threshold means duplicate and will be dropped.
    """
    existing_contents = [str((x or {}).get("content") or "").strip() for x in topic_pool_snippets]
    kept: list[dict[str, Any]] = []

    for sn in new_snippets:
        content = str((sn or {}).get("content") or "").strip()
        if not content:
            continue

        is_dup = False
        # Compare against existing pool first.
        for old_content in existing_contents:
            if _jaccard_similarity_by_content(content, old_content) > threshold:
                is_dup = True
                break
        if is_dup:
            continue

        # Compare against snippets already kept in this same round.
        for accepted in kept:
            if _jaccard_similarity_by_content(content, str(accepted.get("content") or "")) > threshold:
                is_dup = True
                break
        if is_dup:
            continue

        kept_row = dict(sn or {})
        kept_row["url"] = str((sn or {}).get("url") or "").strip()
        kept_row["title"] = str((sn or {}).get("title") or "").strip()
        kept_row["content"] = content
        kept.append(kept_row)

    return kept


def _jaccard_similarity_by_content(a: str, b: str) -> float:
    """Compute content-level Jaccard similarity for deduplication."""
    lang = "zh" if detect_text_language_simple(a + b) == "zh" else "en"
    set_a = set(tokenize_for_bm25(a, lang))
    set_b = set(tokenize_for_bm25(b, lang))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
