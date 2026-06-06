from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


QuestionKey = str  # pool key; currently equals the query string

EN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "their",
    }
)


@dataclass
class QuestionPoolMeta:
    """Metadata for one question-level evidence pool (query + which samples touched it).

    Example::

        QuestionPoolMeta(
            query="When was NVIDIA founded?",
            sample_ids=["sample_001", "sample_002"],
        )
    """

    query: str
    sample_ids: list[str] = field(default_factory=list)


@dataclass
class ClaimSnippetStatus:
    """Per-claim state for multi-round evidence retrieval and final verification.

    Holds claim context, selected snippets, round logs, and the final answer.
    Used by the retrieval pipeline to update one claim across DDG search rounds.

    Example ``to_json_record()`` output (abbreviated)::

        {
            "sample_id": "sample_001",
            "query": "When was NVIDIA founded?",
            "model_sequence_id": 1,
            "atomic_claim_sequence_number": 2,
            "claim_text": "NVIDIA was founded in 1993.",
            "claim_status": "done",
            "final_answer": "supported",
            "evidence_snippet_ids": ["S1", "S3"],
            "round_logs": [{"round_index": 1, "search_query": "NVIDIA founding year", ...}],
        }
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
        """Begin retrieval for this claim using BM25-picked snippets from the topic pool.

        Mutates:
            ``claim_status`` → ``"retrieving"``
            ``selected_initial_snippets`` → copy of input
            ``selected_present_snippets`` → same as initial (starting context for round 1)

        Example before: ``claim_status == "pending"``, both snippet lists empty.

        Example after (input was one snippet)::

            claim_status == "retrieving"
            selected_initial_snippets == [
                {"snippet_id": "S1", "url": "https://...", "title": "...", "content": "NVIDIA was founded in 1993."},
            ]
            selected_present_snippets == selected_initial_snippets  # same list content
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
        extra_trace_fields: dict[str, Any] | None = None,
    ) -> None:
        """Record one completed DDG search + filter round for traceability.

        Mutates:
            ``retrieval_round`` → incremented by 1
            ``searched_queries`` → appends ``search_query``
            ``round_logs`` → appends one dict for this round

        Does not change ``selected_present_snippets`` (call ``update_present_snippets`` separately).

        Example after first round::

            retrieval_round == 1
            searched_queries == ["NVIDIA founding year"]
            round_logs[-1] == {
                "round_index": 1,
                "search_query": "NVIDIA founding year",
                "missing_fact_to_verify": "exact founding date",
                "selected_present_snippet_ids": ["S1", "S2"],
                "newly_added_pool_snippet_ids": ["S5"],
                "direct_relevant_snippet_ids": ["S5"],
                "round_source": "ddg_search",
                "round_outcome": "continue",
            }
        """
        self.retrieval_round += 1
        self.searched_queries.append(search_query)
        row = {
            "round_index": self.retrieval_round,
            "search_query": search_query,
            "missing_fact_to_verify": str(missing_fact_to_verify or "").strip(),
            "selected_present_snippet_ids": list(selected_present_snippet_ids),
            "newly_added_pool_snippet_ids": list(newly_added_pool_snippet_ids),
            "direct_relevant_snippet_ids": list(direct_relevant_snippet_ids),
            "round_source": str(round_source or "").strip(),
            "round_outcome": str(round_outcome or "").strip(),
        }
        if extra_trace_fields:
            row.update(dict(extra_trace_fields))
        self.round_logs.append(row)

    def update_present_snippets(self, present_snippets: list[dict[str, Any]]) -> None:
        """Set which snippets the LLM sees on the next search-or-answer decision.

        Mutates:
            ``selected_present_snippets`` only (``selected_initial_snippets`` unchanged).

        Example before: ``selected_present_snippets == [S1, S2]``

        Example after (LLM filter kept S1 and new S5)::

            selected_present_snippets == [
                {"snippet_id": "S1", "content": "NVIDIA was founded in 1993.", ...},
                {"snippet_id": "S5", "content": "Founded in April 1993.", ...},
            ]
        """
        self.selected_present_snippets = [dict(x) for x in present_snippets]

    def finalize_answer(
        self,
        *,
        final_answer: str,
        final_reason: str,
        evidence_snippet_ids: list[str],
        finished_by_max_round: bool,
    ) -> None:
        """Close retrieval for this claim with a final verification label.

        Mutates:
            ``claim_status`` → ``"done"``
            ``final_answer``, ``final_reason`` → verdict strings
            ``evidence_snippet_ids`` → supporting pool ids (e.g. ``["S1", "S3"]``)
            ``finished_by_max_round`` → True if stopped only because max rounds hit

        Example after call::

            claim_status == "done"
            final_answer == "supported"
            final_reason == "Snippet S1 states the founding year matches the claim."
            evidence_snippet_ids == ["S1", "S3"]
            finished_by_max_round == False
        """
        self.claim_status = "done"
        self.final_answer = str(final_answer or "").strip()
        self.final_reason = str(final_reason or "").strip()
        self.evidence_snippet_ids = [str(x) for x in evidence_snippet_ids if str(x).strip()]
        self.finished_by_max_round = bool(finished_by_max_round)

    def set_error(self, error_text: str) -> None:
        """Attach a failure reason for this claim (does not change ``claim_status`` by itself).

        Mutates:
            ``error`` → stripped error string

        Example after ``set_error("ddg_search_error: timeout")``::

            error == "ddg_search_error: timeout"
            claim_status  # unchanged, e.g. still "retrieving" unless caller sets it
        """
        self.error = str(error_text or "").strip()

    def to_json_record(self) -> dict[str, Any]:
        """Serialize full claim retrieval state for ``claim_status.jsonl``.

        Returns one flat dict (all list fields are copies). See class docstring for shape.
        """
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
    """In-memory registry of question-level evidence pools (shared by query string).

    Stores web snippets per question, assigns stable ``S1``, ``S2`` ids, and picks
    initial snippets for new claims (BM25). Does not run search or LLM policies.

    Internal shape after use::

        _pool_by_question["When was NVIDIA founded?"] = [
            {"snippet_id": "S1", "url": "...", "title": "...", "content": "..."},
        ]
    """

    def __init__(self) -> None:
        """Initialize empty in-memory storage for all question evidence pools.

        Sets state to::

            _pool_by_question == {}
            _meta_by_question == {}
            _next_snippet_id_by_question == {}
        """
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
        """Get or create the pool for ``query``; track ``sample_id`` in metadata.

        Pool is keyed only by ``query`` so snippets reuse across samples on the same question.

        Returns the question key (same as ``query``). Example::

            "When was NVIDIA founded?"
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
        """Return all snippets in one question pool (shallow copy of each row).

        Example return::

            [
                {"snippet_id": "S1", "url": "https://nvidia.com/...", "title": "About", "content": "..."},
                {"snippet_id": "S2", "url": "https://...", "title": "...", "content": "..."},
            ]

        Returns ``[]`` if the pool does not exist or is empty.
        """
        return [dict(x) for x in self._pool_by_question.get(question_key, [])]

    def add_snippets_to_question_pool(
        self,
        question_key: QuestionKey,
        snippets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append new snippets to the pool; assign ``S{n}`` ids when missing.

        Skips blank content and duplicate ``snippet_id`` already in the pool.
        Caller should pre-dedupe by content via ``filter_non_duplicate_snippets_by_content``.

        Returns only the rows actually added. Example return::

            [
                {"snippet_id": "S3", "url": "https://...", "title": "Wiki", "content": "Founded in 1993."},
            ]
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
        remove_query_stopwords: bool = False,
    ) -> list[dict[str, Any]]:
        """Pick starting snippets from the topic pool before the first retrieval round.

        With ``use_bm25=True``, returns BM25 top-k; otherwise returns the full pool copy.

        Example return (top 2)::

            [
                {"snippet_id": "S1", "content": "NVIDIA was founded in April 1993.", ...},
                {"snippet_id": "S4", "content": "The company started as NVISION.", ...},
            ]

        Returns ``[]`` when the pool is empty.
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
            remove_query_stopwords=remove_query_stopwords,
        )

    def dump_question_pool_rows(self) -> list[dict[str, Any]]:
        """Export every in-memory question pool for ``question_evidence_pool.jsonl``.

        Example return (one row per query)::

            [
                {
                    "sample_id": "sample_001",
                    "sample_ids": ["sample_001", "sample_002"],
                    "query": "When was NVIDIA founded?",
                    "pool_snippets": [
                        {"snippet_id": "S1", "url": "...", "title": "...", "content": "..."},
                    ],
                },
            ]
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
    """Detect zh vs en for BM25 tokenization (CJK present → ``zh``, else ``en``).

    Returns ``"zh"`` or ``"en"``. Example: ``detect_text_language_simple("成立于1993")`` → ``"zh"``.
    """
    t = str(text or "")
    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    return "en"


def tokenize_for_bm25(
    text: str,
    lang: str,
    *,
    for_query: bool = False,
    remove_stopwords: bool = False,
) -> list[str]:
    """Tokenize text for BM25 (English words; Chinese chars + bigrams + embedded English).

    Returns a token list. Example::

        tokenize_for_bm25("NVIDIA was founded in 1993", "en")
        # -> ["nvidia", "was", "founded", "in", "1993"]
    """
    s = str(text or "").lower()
    if not s:
        return []

    if lang == "zh":
        zh_chars = re.findall(r"[\u4e00-\u9fff]", s)
        zh_bigrams = ["".join(zh_chars[i : i + 2]) for i in range(max(0, len(zh_chars) - 1))]
        en_words = re.findall(r"[a-z0-9]+", s)
        return zh_chars + zh_bigrams + en_words

    tokens = re.findall(r"[a-z0-9]+", s)
    if for_query and remove_stopwords:
        tokens = [t for t in tokens if t not in EN_STOPWORDS]
    return tokens


def rank_snippets_by_bm25(
    claim_text: str,
    snippets: list[dict[str, Any]],
    top_k: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    remove_query_stopwords: bool = False,
) -> list[dict[str, Any]]:
    """Rank pool snippets by BM25 against ``claim_text``; return top-k snippet dicts (copies).

    Example: with 10 pool snippets and ``top_k=3``, returns the 3 highest-scoring rows
    (same keys as input, e.g. ``snippet_id``, ``content``). Returns ``[]`` if ``top_k <= 0``
    or query tokenizes to empty.
    """
    if top_k <= 0:
        return []

    docs: list[list[str]] = []
    for sn in snippets:
        content = str((sn or {}).get("content") or "").strip()
        lang = detect_text_language_simple(content)
        docs.append(tokenize_for_bm25(content, lang))

    query_lang = detect_text_language_simple(claim_text)
    query_terms = tokenize_for_bm25(
        claim_text,
        query_lang,
        for_query=True,
        remove_stopwords=remove_query_stopwords,
    )
    if not query_terms:
        return []

    n_docs = len(docs)
    if n_docs == 0:
        return []

    avgdl = sum(len(d) for d in docs) / n_docs if n_docs > 0 else 0.0
    avgdl = max(avgdl, 1.0)

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
    """Drop near-duplicate snippets by Jaccard similarity on ``content`` only.

    Compares against the existing pool and snippets already accepted in this batch.

    Example: pool has ``"Founded in 1993."``; new snippet with ~same content is dropped.
    Returns kept rows (may be empty) with normalized ``url``, ``title``, ``content``::

        [{"url": "https://...", "title": "Wiki", "content": "NVISION was the early name."}]
    """
    # Precompute token sets once so the dedup pass stays linear-ish even when the
    # topic pool becomes large.
    existing_token_sets = [
        _content_token_set_for_dedup(str((x or {}).get("content") or "").strip()) for x in topic_pool_snippets
    ]
    kept: list[dict[str, Any]] = []
    kept_token_sets: list[set[str]] = []

    for sn in new_snippets:
        content = str((sn or {}).get("content") or "").strip()
        if not content:
            continue

        token_set = _content_token_set_for_dedup(content)
        is_dup = False
        # First compare against snippets that are already in the topic pool.
        for old_token_set in existing_token_sets:
            if _jaccard_similarity_by_token_set(token_set, old_token_set) > threshold:
                is_dup = True
                break
        if is_dup:
            continue

        # Then compare against snippets accepted earlier in the same batch so
        # near-duplicates do not enter together.
        for accepted_token_set in kept_token_sets:
            if _jaccard_similarity_by_token_set(token_set, accepted_token_set) > threshold:
                is_dup = True
                break
        if is_dup:
            continue

        kept_row = dict(sn or {})
        kept_row["url"] = str((sn or {}).get("url") or "").strip()
        kept_row["title"] = str((sn or {}).get("title") or "").strip()
        kept_row["content"] = content
        kept.append(kept_row)
        kept_token_sets.append(token_set)

    return kept


def find_best_pool_duplicate_by_content(
    new_snippet: dict[str, Any],
    topic_pool_snippets: list[dict[str, Any]],
    threshold: float = 0.9,
) -> dict[str, Any] | None:
    """Return the most similar existing pool snippet when similarity exceeds threshold."""
    content = str((new_snippet or {}).get("content") or "").strip()
    if not content:
        return None
    token_set = _content_token_set_for_dedup(content)
    best_match: dict[str, Any] | None = None
    best_score = threshold
    for old in topic_pool_snippets:
        old_content = str((old or {}).get("content") or "").strip()
        if not old_content:
            continue
        score = _jaccard_similarity_by_token_set(
            token_set,
            _content_token_set_for_dedup(old_content),
        )
        if score > best_score:
            best_score = score
            best_match = dict(old)
    return best_match


def _jaccard_similarity_by_content(a: str, b: str) -> float:
    """Jaccard similarity of token sets from two content strings (0.0–1.0)."""
    set_a = _content_token_set_for_dedup(a)
    set_b = _content_token_set_for_dedup(b)
    return _jaccard_similarity_by_token_set(set_a, set_b)


def _content_token_set_for_dedup(text: str) -> set[str]:
    """Token set for content dedup; English removes stopwords before Jaccard.

    This is intentionally separate from BM25 scoring. We still use a token-set
    Jaccard comparison here; the only change is the token view used to build the
    set.
    """
    s = str(text or "").strip()
    if not s:
        return set()

    lang = "zh" if detect_text_language_simple(s) == "zh" else "en"
    tokens = tokenize_for_bm25(s, lang)
    if lang == "en":
        tokens = [t for t in tokens if t not in EN_STOPWORDS]
    return set(tokens)


def _jaccard_similarity_by_token_set(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity of two token sets (0.0–1.0)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
