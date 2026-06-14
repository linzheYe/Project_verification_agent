from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import requests

from scripts.data_io import parse_json_object_robust
from scripts.ddg_search import search_snippets_with_ddg
from scripts.evidence_registry import _content_token_set_for_dedup, _jaccard_similarity_by_token_set
from scripts.llm_api import LLMClient, LLMConnectivityError
from scripts.prompt_template import (
    build_topic_grounding_evidence_extract_prompt,
    build_topic_grounding_query_plan_prompt,
    build_topic_grounding_url_select_prompt,
)


@dataclass(frozen=True)
class PrefetchStageConfig:
    """Config for topic prefetch stages."""

    wiki_query_count: int = 3
    web_query_count: int = 3
    snippets_per_query: int = 8
    max_selected_candidates: int = 8
    fetch_timeout_seconds: int = 20
    max_page_text_chars: int = 30000
    near_dup_threshold: float = 0.9
    fetch_retry_count: int = 1


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    query_text: str


@dataclass(frozen=True)
class SearchSnippetCandidate:
    candidate_id: str
    topic_id: str
    query_id: str
    query_text: str
    url: str
    title: str
    snippet: str
    rank: int


@dataclass(frozen=True)
class SelectedSearchUrl:
    topic_id: str
    url_id: str
    url: str
    source_title: str
    candidate_ids: list[str]
    query_ids: list[str]


@dataclass(frozen=True)
class FetchedPage:
    topic_id: str
    query: str
    url_id: str
    source_url: str
    source_title: str
    source_type: str  # external | search
    candidate_ids: list[str]
    query_ids: list[str]
    fetch_error: str | None
    full_text: str


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_url(url: str) -> str:
    normalized = str(url or "").strip()
    normalized = re.sub(r"#.*$", "", normalized)
    return normalized.rstrip("/")


SNIPPET_SEGMENT_SEPARATOR = "\n\n......\n\n"

def _jaccard_similarity(text_a: str, text_b: str) -> float:
    return _jaccard_similarity_by_token_set(
        _content_token_set_for_dedup(text_a),
        _content_token_set_for_dedup(text_b),
    )


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


# =============================
# Stage 1: plan query + search
# =============================
def stage1_plan_and_search(
    *,
    llm: LLMClient,
    topic_id: str,
    query: str,
    claims: list[dict[str, Any]],
    config: PrefetchStageConfig,
) -> tuple[list[SearchQuery], list[SearchSnippetCandidate]]:
    """Generate retrieval queries and retrieve raw DDG snippet candidates."""
    search_queries = _generate_search_queries_with_llm(llm=llm, query=query, claims=claims, config=config)
    search_candidates = _retrieve_ddg_snippet_candidates(topic_id=topic_id, search_queries=search_queries, config=config)
    return search_queries, search_candidates


def _generate_search_queries_with_llm(
    *, llm: LLMClient, query: str, claims: list[dict[str, Any]], config: PrefetchStageConfig
) -> list[SearchQuery]:
    system_prompt, user_prompt = build_topic_grounding_query_plan_prompt(
        {
            "query": query,
            "claims": claims,
            "background_text": "",
            "wiki_query_count": config.wiki_query_count,
            "web_query_count": config.web_query_count,
        }
    )
    llm_json = _call_llm_json(llm, system_prompt, user_prompt, stage_name="stage1_query_plan")

    wiki_queries = [_normalize_whitespace(item) for item in (llm_json.get("wiki_queries") or []) if _normalize_whitespace(item)]
    web_queries = [_normalize_whitespace(item) for item in (llm_json.get("web_queries") or []) if _normalize_whitespace(item)]

    wiki_queries = wiki_queries[: config.wiki_query_count]
    web_queries = web_queries[: config.web_query_count]

    while len(wiki_queries) < config.wiki_query_count:
        wiki_queries.append(f"{query} wiki")
    while len(web_queries) < config.web_query_count:
        web_queries.append(query)

    result: list[SearchQuery] = []
    for index, query_text in enumerate(wiki_queries, start=1):
        result.append(SearchQuery(query_id=f"QW{index}", query_text=query_text))
    for index, query_text in enumerate(web_queries, start=1):
        result.append(SearchQuery(query_id=f"QG{index}", query_text=query_text))
    return result


def _retrieve_ddg_snippet_candidates(
    *, topic_id: str, search_queries: list[SearchQuery], config: PrefetchStageConfig
) -> list[SearchSnippetCandidate]:
    candidates: list[SearchSnippetCandidate] = []
    for search_query in search_queries:
        try:
            snippet_rows = search_snippets_with_ddg(search_query.query_text, max_results=config.snippets_per_query)
        except Exception:
            snippet_rows = []

        for rank, snippet_row in enumerate(snippet_rows, start=1):
            candidates.append(
                SearchSnippetCandidate(
                    candidate_id=f"{topic_id}_{search_query.query_id}_S{rank}",
                    topic_id=topic_id,
                    query_id=search_query.query_id,
                    query_text=search_query.query_text,
                    url=normalize_url(snippet_row.get("url") or ""),
                    title=_normalize_whitespace(snippet_row.get("title") or ""),
                    snippet=_normalize_whitespace(snippet_row.get("content") or ""),
                    rank=rank,
                )
            )
    return candidates


# ==================================
# Shared candidate preparation logic
# ==================================
def dedup_search_candidates_by_url(candidates: list[SearchSnippetCandidate]) -> list[SearchSnippetCandidate]:
    """Merge candidates that share the same normalized URL."""
    merged_by_url: dict[str, dict[str, Any]] = {}
    url_order: list[str] = []
    for candidate in candidates:
        if not candidate.url:
            continue
        merged = merged_by_url.get(candidate.url)
        if merged is None:
            merged_by_url[candidate.url] = {
                "topic_id": candidate.topic_id,
                "url": candidate.url,
                "title": candidate.title,
                "snippets": [candidate.snippet] if candidate.snippet else [],
            }
            url_order.append(candidate.url)
            continue
        if not merged.get("title") and candidate.title:
            merged["title"] = candidate.title
        if candidate.snippet and candidate.snippet not in merged["snippets"]:
            merged["snippets"].append(candidate.snippet)

    deduplicated: list[SearchSnippetCandidate] = []
    for index, url in enumerate(url_order, start=1):
        merged = merged_by_url[url]
        combined_snippet = SNIPPET_SEGMENT_SEPARATOR.join(
            snippet for snippet in merged.get("snippets", []) if _normalize_whitespace(snippet)
        ).strip()
        if not combined_snippet:
            continue
        topic_id = str(merged.get("topic_id") or "T0")
        deduplicated.append(
            SearchSnippetCandidate(
                candidate_id=f"{topic_id}_TC{index}",
                topic_id=topic_id,
                query_id="",
                query_text="",
                url=url,
                title=str(merged.get("title") or ""),
                snippet=combined_snippet,
                rank=index,
            )
        )
    return deduplicated


def dedup_search_candidates_by_snippet_similarity(
    candidates: list[SearchSnippetCandidate], near_dup_threshold: float
) -> list[SearchSnippetCandidate]:
    """Dedup by snippet content overlap using stopword-aware Jaccard similarity."""
    deduplicated: list[SearchSnippetCandidate] = []
    for candidate in candidates:
        if not candidate.url or not candidate.snippet:
            continue

        is_duplicate = False
        for kept in deduplicated:
            if _jaccard_similarity(kept.snippet, candidate.snippet) >= near_dup_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(candidate)
    return deduplicated


def remove_candidates_that_overlap_external_urls(
    candidates: list[SearchSnippetCandidate], external_urls: list[str]
) -> list[SearchSnippetCandidate]:
    """Remove search candidates whose URL already exists in external URLs."""
    external_url_set = {normalize_url(url) for url in external_urls if normalize_url(url)}
    if not external_url_set:
        return list(candidates)
    return [candidate for candidate in candidates if candidate.url not in external_url_set]


def prepare_search_candidates_for_selection(
    *,
    raw_search_candidates: list[SearchSnippetCandidate],
    external_urls: list[str],
    config: PrefetchStageConfig,
) -> list[SearchSnippetCandidate]:
    """Prepare clean search candidate pool for LLM selection.

    Order matters:
    1) URL-level dedup
    2) snippet-level near-dup dedup
    3) remove URLs already covered by trusted external URLs
    """
    url_deduped = dedup_search_candidates_by_url(raw_search_candidates)
    snippet_deduped = dedup_search_candidates_by_snippet_similarity(url_deduped, near_dup_threshold=config.near_dup_threshold)
    return remove_candidates_that_overlap_external_urls(snippet_deduped, external_urls=external_urls)


# =========================================
# Stage 2: fetch trusted external URLs first
# =========================================
def stage2_prepare_external_fulltext(
    *,
    topic_id: str,
    query: str,
    external_urls: list[str],
    config: PrefetchStageConfig,
) -> list[FetchedPage]:
    """Fetch fulltext for trusted external URLs directly (no snippet selection)."""
    fetched_pages: list[FetchedPage] = []
    for index, raw_url in enumerate(external_urls, start=1):
        normalized_url = normalize_url(raw_url)
        if not normalized_url:
            continue
        full_text, fetch_error = _fetch_fulltext_with_retry(
            normalized_url,
            timeout_seconds=config.fetch_timeout_seconds,
            max_page_text_chars=config.max_page_text_chars,
            retry_count=config.fetch_retry_count,
        )
        fetched_pages.append(
            FetchedPage(
                topic_id=topic_id,
                query=query,
                url_id=f"{topic_id}_EXT_U{index}",
                source_url=normalized_url,
                source_title="external_url",
                source_type="external",
                candidate_ids=[],
                query_ids=["QEXT"],
                fetch_error=fetch_error,
                full_text=full_text,
            )
        )
    return fetched_pages


# ============================================
# Stage 3: LLM selection only on search result
# ============================================
def stage3_select_search_urls(
    *,
    llm: LLMClient,
    query: str,
    claims: list[dict[str, Any]],
    prepared_search_candidates: list[SearchSnippetCandidate],
    config: PrefetchStageConfig,
) -> list[SelectedSearchUrl]:
    """Select candidate search URLs with LLM snippet filtering."""
    if not prepared_search_candidates:
        return []

    prompt_candidates = [
        {
            "candidate_id": candidate.candidate_id,
            "url": candidate.url,
            "title": candidate.title,
            "content": candidate.snippet,
        }
        for candidate in prepared_search_candidates
    ]

    system_prompt, user_prompt = build_topic_grounding_url_select_prompt(
        {
            "query": query,
            "claims": claims,
            "background_text": "",
            "candidates": prompt_candidates,
            "max_selected_candidates": config.max_selected_candidates,
        }
    )
    llm_json = _call_llm_json(llm, system_prompt, user_prompt, stage_name="stage3_select_search_urls")

    picked_candidate_ids = [_normalize_whitespace(x) for x in (llm_json.get("selected_candidate_ids") or []) if _normalize_whitespace(x)]
    valid_candidate_ids = {candidate.candidate_id for candidate in prepared_search_candidates}

    selected_candidate_ids: list[str] = []
    selected_id_set: set[str] = set()
    for candidate_id in picked_candidate_ids:
        if candidate_id in valid_candidate_ids and candidate_id not in selected_id_set:
            selected_candidate_ids.append(candidate_id)
            selected_id_set.add(candidate_id)
        if len(selected_candidate_ids) >= config.max_selected_candidates:
            break

    if not selected_candidate_ids:
        # Conservative fallback when LLM returns nothing.
        selected_candidate_ids = [
            candidate.candidate_id
            for candidate in prepared_search_candidates
            if "wikipedia.org" in candidate.url.lower()
        ][: min(2, config.max_selected_candidates)]

    selected_candidates = [candidate for candidate in prepared_search_candidates if candidate.candidate_id in set(selected_candidate_ids)]

    url_to_meta: dict[str, dict[str, Any]] = {}
    topic_id = selected_candidates[0].topic_id if selected_candidates else "T0"
    for candidate in selected_candidates:
        if candidate.url not in url_to_meta:
            url_to_meta[candidate.url] = {
                "source_title": candidate.title,
                "candidate_ids": [],
                "query_ids": [],
            }
        url_to_meta[candidate.url]["candidate_ids"].append(candidate.candidate_id)
        if candidate.query_id not in url_to_meta[candidate.url]["query_ids"]:
            url_to_meta[candidate.url]["query_ids"].append(candidate.query_id)

    selected_urls: list[SelectedSearchUrl] = []
    for index, (url, meta) in enumerate(url_to_meta.items(), start=1):
        selected_urls.append(
            SelectedSearchUrl(
                topic_id=topic_id,
                url_id=f"{topic_id}_SEL_U{index}",
                url=url,
                source_title=str(meta.get("source_title") or ""),
                candidate_ids=list(meta.get("candidate_ids") or []),
                query_ids=list(meta.get("query_ids") or []),
            )
        )
    return selected_urls


# ======================================================
# Stage 4: fetch selected search URLs and merge evidence
# ======================================================
def stage4_fetch_selected_search_fulltext(
    *,
    topic_id: str,
    query: str,
    selected_search_urls: list[SelectedSearchUrl],
    config: PrefetchStageConfig,
) -> list[FetchedPage]:
    """Fetch fulltext for LLM-selected search URLs."""
    fetched_pages: list[FetchedPage] = []
    for selected_url in selected_search_urls:
        full_text, fetch_error = _fetch_fulltext_with_retry(
            selected_url.url,
            timeout_seconds=config.fetch_timeout_seconds,
            max_page_text_chars=config.max_page_text_chars,
            retry_count=config.fetch_retry_count,
        )
        fetched_pages.append(
            FetchedPage(
                topic_id=topic_id,
                query=query,
                url_id=selected_url.url_id,
                source_url=selected_url.url,
                source_title=selected_url.source_title,
                source_type="search",
                candidate_ids=list(selected_url.candidate_ids),
                query_ids=list(selected_url.query_ids),
                fetch_error=fetch_error,
                full_text=full_text,
            )
        )
    return fetched_pages


def merge_fetched_pages_external_first(
    external_fetched_pages: list[FetchedPage],
    selected_search_fetched_pages: list[FetchedPage],
) -> list[FetchedPage]:
    """Merge fetched pages by URL, preferring external pages on conflicts."""
    merged: list[FetchedPage] = []
    seen_urls: set[str] = set()

    for page in external_fetched_pages + selected_search_fetched_pages:
        normalized = normalize_url(page.source_url)
        if not normalized or normalized in seen_urls:
            continue
        seen_urls.add(normalized)
        merged.append(page)
    return merged


# ======================================
# Stage 5: evidence extraction with LLM
# ======================================
def stage5_extract_evidence(
    *,
    llm: LLMClient,
    topic_id: str,
    query: str,
    claims: list[dict[str, Any]],
    merged_fetched_pages: list[FetchedPage],
) -> tuple[list[dict[str, Any]], str]:
    """Extract evidence items and topic brief from merged fetched pages."""
    usable_pages = [page for page in merged_fetched_pages if not page.fetch_error and page.full_text.strip()]
    if not usable_pages:
        return [], ""

    pages_for_prompt = [
        {
            "url_id": page.url_id,
            "topic_id": page.topic_id,
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

    system_prompt, user_prompt = build_topic_grounding_evidence_extract_prompt(
        {
            "topic_id": topic_id,
            "query": query,
            "claims": claims,
            "background_text": "",
            "pages": pages_for_prompt,
        }
    )
    llm_json = _call_llm_json(llm, system_prompt, user_prompt, stage_name="stage5_extract_evidence")

    valid_url_ids = {page.url_id for page in usable_pages}
    evidence_items: list[dict[str, Any]] = []
    evidence_index = 0

    for raw_item in (llm_json.get("evidence_items") or []):
        if not isinstance(raw_item, dict):
            continue

        url_id = _normalize_whitespace(raw_item.get("url_id") or "")
        relevant_text = str(raw_item.get("relevant_text") or "").strip()
        summary = _normalize_whitespace(raw_item.get("summary") or "")

        if not url_id or url_id not in valid_url_ids or not relevant_text or not summary:
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
                "relevant_text": relevant_text,
                "summary": summary,
                "content": f"{relevant_text}\n\nSummary: {summary}".strip(),
            }
        )

    topic_brief = _normalize_whitespace(llm_json.get("topic_brief") or "")
    return evidence_items, topic_brief


# ======================
# Shared fetch utilities
# ======================
def _clean_extracted_text(text: str) -> str:
    text = str(text or "").replace("\u00a0", " ").replace("\ufeff", "")
    lines = [line.strip() for line in text.splitlines()]
    drop_patterns = [
        r"^(cookie|cookies|隐私|privacy policy|版权所有|copyright)\\b",
        r"^(登录|注册|订阅|分享|返回顶部)$",
    ]
    kept_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        lower_line = line.lower()
        if any(re.search(pattern, lower_line) for pattern in drop_patterns):
            continue
        kept_lines.append(_normalize_whitespace(line))
    return "\n".join(kept_lines).strip()


def _fetch_fulltext_once(url: str, timeout_seconds: int, max_page_text_chars: int) -> tuple[str, str | None]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout_seconds, allow_redirects=True)
    except Exception as exc:
        return "", f"requests_error:{type(exc).__name__}"

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "application/pdf" in content_type or str(url).lower().endswith(".pdf"):
        return "", "unsupported_pdf"
    if response.status_code >= 400:
        return "", f"http_{response.status_code}"

    html = str(response.text or "")
    if not html.strip():
        return "", "empty_response"

    extracted_text = ""
    extract_error: str | None = None
    try:
        import trafilatura  # type: ignore

        extracted_text = str(
            trafilatura.extract(
                html,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
            or ""
        )
    except Exception as exc:
        extract_error = f"trafilatura_error:{type(exc).__name__}"

    if not extracted_text.strip():
        fallback_text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        fallback_text = re.sub(r"<style[\s\S]*?</style>", " ", fallback_text, flags=re.IGNORECASE)
        fallback_text = re.sub(r"<[^>]+>", " ", fallback_text)
        extracted_text = _normalize_whitespace(fallback_text)

    cleaned_text = _clean_extracted_text(extracted_text)
    if len(cleaned_text) > max_page_text_chars:
        cleaned_text = cleaned_text[:max_page_text_chars]

    if not cleaned_text.strip():
        return "", extract_error or "extract_empty"
    return cleaned_text, None


def _fetch_fulltext_with_retry(
    url: str,
    *,
    timeout_seconds: int,
    max_page_text_chars: int,
    retry_count: int,
) -> tuple[str, str | None]:
    """Fetch fulltext with simple retry policy for transient fetch failures."""
    final_text = ""
    final_error: str | None = None
    attempts = max(0, int(retry_count)) + 1

    for _ in range(attempts):
        final_text, final_error = _fetch_fulltext_once(url, timeout_seconds, max_page_text_chars)
        if final_text.strip() and not final_error:
            return final_text, None

    return final_text, final_error


def as_jsonable_rows(items: list[Any]) -> list[dict[str, Any]]:
    """Convert dataclass rows to JSON-serializable dict rows."""
    return [asdict(item) for item in items]
