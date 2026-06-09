from scripts.prefetch_topic_evidence_stages import (
    PrefetchStageConfig,
    SearchSnippetCandidate,
    dedup_search_candidates_by_snippet_similarity,
    dedup_search_candidates_by_url,
    prepare_search_candidates_for_selection,
    stage5_extract_evidence,
)


class DummyLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def call(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        import json

        return json.dumps(self.payload, ensure_ascii=False)


def test_dedup_search_candidates_by_url_merges_snippets_and_reassigns_ids() -> None:
    candidates = [
        SearchSnippetCandidate(
            candidate_id="T1_QW1_S1",
            topic_id="T1",
            query_id="QW1",
            query_text="alpha wiki",
            url="https://example.com/page",
            title="Example Page",
            snippet="First summary.",
            rank=1,
        ),
        SearchSnippetCandidate(
            candidate_id="T1_QG1_S2",
            topic_id="T1",
            query_id="QG1",
            query_text="alpha details",
            url="https://example.com/page",
            title="",
            snippet="Second summary.",
            rank=2,
        ),
    ]

    merged = dedup_search_candidates_by_url(candidates)

    assert len(merged) == 1
    assert merged[0].candidate_id == "T1_TC1"
    assert merged[0].title == "Example Page"
    assert merged[0].snippet == "First summary.\n\nSecond summary."


def test_dedup_search_candidates_by_snippet_similarity_uses_content_only() -> None:
    candidates = [
        SearchSnippetCandidate(
            candidate_id="T1_TC1",
            topic_id="T1",
            query_id="",
            query_text="",
            url="https://example.com/a",
            title="Alpha Title",
            snippet="The company was founded in 1993 in California.",
            rank=1,
        ),
        SearchSnippetCandidate(
            candidate_id="T1_TC2",
            topic_id="T1",
            query_id="",
            query_text="",
            url="https://example.com/b",
            title="Completely Different Heading",
            snippet="Company founded in 1993 in California.",
            rank=2,
        ),
    ]

    deduped = dedup_search_candidates_by_snippet_similarity(candidates, near_dup_threshold=0.9)

    assert len(deduped) == 1
    assert deduped[0].candidate_id == "T1_TC1"


def test_prepare_search_candidates_for_selection_filters_external_urls_after_merge() -> None:
    raw_candidates = [
        SearchSnippetCandidate(
            candidate_id="T2_QW1_S1",
            topic_id="T2",
            query_id="QW1",
            query_text="topic wiki",
            url="https://example.com/keep",
            title="Keep",
            snippet="Keep this snippet.",
            rank=1,
        ),
        SearchSnippetCandidate(
            candidate_id="T2_QW2_S1",
            topic_id="T2",
            query_id="QW2",
            query_text="topic wiki alt",
            url="https://example.com/drop",
            title="Drop",
            snippet="Drop this snippet.",
            rank=1,
        ),
        SearchSnippetCandidate(
            candidate_id="T2_QG1_S1",
            topic_id="T2",
            query_id="QG1",
            query_text="topic details",
            url="https://example.com/drop",
            title="Drop",
            snippet="Another angle for the same page.",
            rank=2,
        ),
    ]

    prepared = prepare_search_candidates_for_selection(
        raw_search_candidates=raw_candidates,
        external_urls=["https://example.com/drop/"],
        config=PrefetchStageConfig(),
    )

    assert [candidate.candidate_id for candidate in prepared] == ["T2_TC1"]
    assert prepared[0].url == "https://example.com/keep"


def test_stage5_extract_evidence_keeps_formal_snippet_ids_independent_from_candidate_ids() -> None:
    llm = DummyLLM(
        {
            "evidence_items": [
                {
                    "url_id": "T3_SEL_U1",
                    "relevant_text": "Founded in 1993.",
                    "summary": "The company was founded in 1993.",
                }
            ],
            "topic_brief": "Founded in 1993 [S1].",
        }
    )
    merged_fetched_pages = [
        type(
            "FetchedPageLike",
            (),
            {
                "topic_id": "T3",
                "query": "When was the company founded?",
                "url_id": "T3_SEL_U1",
                "source_url": "https://example.com/founding",
                "source_title": "Founding History",
                "source_type": "search",
                "candidate_ids": ["T3_TC4"],
                "query_ids": [],
                "fetch_error": None,
                "full_text": "Founded in 1993 with supporting details.",
            },
        )()
    ]

    evidence_items, topic_brief = stage5_extract_evidence(
        llm=llm,
        topic_id="T3",
        query="When was the company founded?",
        claims=[{"claim_id": "T3_C1", "claim_text": "When was the company founded?"}],
        merged_fetched_pages=merged_fetched_pages,
    )

    assert [item["snippet_id"] for item in evidence_items] == ["S1"]
    assert evidence_items[0]["url_id"] == "T3_SEL_U1"
    assert topic_brief == "Founded in 1993 [S1]."
