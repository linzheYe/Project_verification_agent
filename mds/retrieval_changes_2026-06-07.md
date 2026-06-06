# Retrieval Changes 2026-06-07

## Overview

This change set updated the claim-level retrieval flow in three places:

1. `SNIPPET_FILTER` now receives `existing_claim_snippets` so the model can avoid adding redundant snippets for the same claim.
2. `NEXT_SEARCH_OR_ANSWER` now receives `search_history` instead of `previous_searched_queries` and `present_snippets_with_ids`, so the model can see which search query produced which newly added direct-evidence snippets.
3. `MUST_HAVE_ANSWER` no longer receives round metadata such as `present_added_round`; it now gets snippet records stripped back to `snippet_id/url/title/content`.

The goal was to make the retrieval loop more incremental and easier for the model to reason about:

- Reduce redundant additions during snippet filtering.
- Preserve direct evidence even when a new direct candidate is deduplicated against the existing evidence pool.
- Show the model a round-by-round mapping from search query to newly added direct evidence, so it can change search strategy more intelligently.
- Keep final must-answer judgment free of retrieval-round metadata.

## Files Changed

- `scripts/prompt_template.py`
- `pipeline/pipeline_evidence_retrieve.py`
- `scripts/evidence_registry.py`

## Detailed Changes

### 1. `SNIPPET_FILTER` prompt input cleanup

The snippet filter prompt no longer relies on topic-level wording. It now uses:

- `current_round_snippets`
- `existing_claim_snippets`

`existing_claim_snippets` is passed in as claim-local prior evidence and contains only:

- `snippet_id`
- `url`
- `title`
- `content`

No round metadata is passed into `SNIPPET_FILTER`.

Implementation details:

- `build_snippet_filter_prompt(...)` now reads `existing_claim_snippets`.
- The retrieval pipeline passes `existing_claim_snippets` using `_snippet_brief_without_round(...)`.
- Stale `topic_block` handling was removed from the snippet-filter builder.

### 2. Direct evidence fallback when pool dedup blocks a new direct candidate

Previously, if a new direct candidate was selected by the LLM but did not enter the evidence pool due to deduplication, it would be lost as direct evidence for the claim.

Now:

- If the direct candidate enters the pool, it is used normally.
- If it is deduplicated against the existing pool, the pipeline finds the most similar existing pool snippet above the duplication threshold and uses that existing pool snippet as the direct evidence replacement.

Implementation details:

- Added `find_best_pool_duplicate_by_content(...)` in `scripts/evidence_registry.py`.
- Integrated this fallback into the direct-snippet construction path in `pipeline_evidence_retrieve.py`.

### 3. Round tagging inside claim-local `present_snippets`

`present_snippets` now carry `present_added_round` for internal retrieval use.

Rules:

- Initial snippets are tagged with `present_added_round = 0`
- Newly added direct snippets are tagged with the current round index
- Query-cache reused direct snippets are retagged with the current round index for the current claim

This metadata is only used where round-aware retrieval context is useful. It is explicitly stripped before passing snippets into prompts that should not see round metadata.

Implementation details:

- Added `_with_present_round(...)`
- Extended `_snippet_brief(...)` to preserve `present_added_round`
- Added `_snippet_brief_without_round(...)` for prompt inputs that should stay clean

### 4. `NEXT_SEARCH_OR_ANSWER` switched to `search_history`

The next-action prompt no longer receives:

- `previous_searched_queries`
- `present_snippets_with_ids`

Instead, it receives:

- `search_history`

Each history item contains:

- `round_index`
- `search_query`
- `newly_added_present_snippet_ids`
- `newly_added_present_snippets`

`newly_added_present_snippet_ids` are built from `direct_relevant_snippet_ids` in `round_logs`.
Only valid evidence-pool ids matching `S\d+` are retained. If an id cannot be resolved to a current claim-visible snippet, it is skipped.

This means the model now sees retrieval history as:

- what was searched
- what direct evidence that search actually added

instead of two disconnected views.

Implementation details:

- Added `_build_search_history_for_next_action(status)`
- `build_next_search_or_answer_prompt(...)` now expects `search_history`
- The retrieval loop now passes `search_history` into `NEXT_SEARCH_OR_ANSWER`

### 5. Query-cache metadata cleanup

Cached direct snippets should not carry claim-specific round tags across claims.

Implementation details:

- `_cache_query_outcome(...)` now strips `present_added_round` before storing `direct_snippets` in the query cache

### 6. `MUST_HAVE_ANSWER` no longer sees round metadata

`MUST_HAVE_ANSWER` should judge evidence content only, not retrieval timing.

Implementation details:

- `_finalize_with_must_answer(...)` now passes `present_snippets_with_ids` through `_snippet_brief_without_round(...)`
- So `build_must_have_answer_prompt(...)` receives only:
  - `snippet_id`
  - `url`
  - `title`
  - `content`

### 7. Restored `last_step_feedback_block` in `NEXT_SEARCH_OR_ANSWER`

During self-check, I found that `build_next_search_or_answer_prompt(...)` was still constructing `last_step_feedback_block`, but the updated user prompt template no longer rendered it.

This was a real bug because failed-query feedback would silently stop reaching the model.

Implementation details:

- Reinserted `{last_step_feedback_block}` into `NEXT_SEARCH_OR_ANSWER_USER_PROMPT_TEMPLATE`

## Self-Check and Verification

I rechecked the modified logic for the three affected prompt paths:

### `SNIPPET_FILTER`

- Receives `existing_claim_snippets`
- Does not receive round metadata
- No stale topic-block formatting remains in the builder

### `NEXT_SEARCH_OR_ANSWER`

- Receives `search_history`
- `search_history` is built only from completed prior rounds
- `newly_added_present_snippet_ids` are restricted to valid pool ids (`S\d+`)
- `newly_added_present_snippets` are resolved from current claim-visible snippets
- Failed-query retry feedback is rendered again via `last_step_feedback_block`

### `MUST_HAVE_ANSWER`

- Does not receive `present_added_round`
- Still receives stable pool snippet ids and full snippet content

## Remaining Notes

- `search_history` intentionally contains both `newly_added_present_snippet_ids` and `newly_added_present_snippets`. This is mildly redundant, but retained on purpose for easier debugging and traceability.
- Internal claim state still keeps `present_added_round`; only prompt payloads that should be round-agnostic strip it out.

## Validation Performed

Ran:

```bash
python -m py_compile scripts/prompt_template.py pipeline/pipeline_evidence_retrieve.py scripts/evidence_registry.py
```

Result:

- Passed without syntax errors.
