from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
# claim prepare: clean split 
# prefetch
# retrive
# rubric
CLEAN_RESPONSE_SYSTEM_PROMPT_TEMPLATE = """
Clean and normalize an LLM response while preserving its factual meaning.

Your task is to remove low-value, non-informative, or formulaic text, and rewrite the remaining content into clear, complete, self-contained statements.

Remove expressions such as:
- Simple confirmations, such as "Yes," "Correct," or "Sure."
- Generic lead-ins, such as "The information is as follows" or "Here are the details."
- Empty transition phrases, such as "Specifically," "In particular," or "It should be noted that."
- Summary fillers, such as "In summary" or "Overall."
- Any other generic wording that does not add factual content.

Rules:
- Preserve all factual information from the input.
- Do not add facts that are not present in the input.
- Do not change dates, numbers, names, titles, entities, or technical terms.
- If the input contains incomplete fragments, labels, headings, or bullet-style fields, rewrite them as complete sentences when the missing sentence structure can be inferred from the local context.
- If a fragment cannot be completed without guessing, keep it concise but do not invent missing information.
- Remove duplicated content.
- Keep the cleaned text in the same language as the input.
- Return only valid JSON.
- Do not add explanations, markdown, comments, or extra fields.

Return the result using exactly this schema:

{
  "cleaned_response": "cleaned response text"
}

Example 1:
Input:
Sure. Here are the details: Tesla reported revenue of $96.8 billion in 2023. Specifically, its automotive revenue was $82.4 billion.
Output:
{
  "cleaned_response": "Tesla reported revenue of $96.8 billion in 2023. Its automotive revenue was $82.4 billion."
}

Example 2:
Input:
Details:
Selection batch: Third batch
Selection date: June 8, 2011
Output:
{
  "cleaned_response": "The item was selected in the third batch, with a selection date of June 8, 2011."
}

Example 3:
Input:
Brief conclusion:
The policy coverage expanded.
Covered entities: small and medium-sized enterprises.
Implementation date: January 2024.
Output:
{
  "cleaned_response": "The policy coverage expanded to include small and medium-sized enterprises, with an implementation date of January 2024."
}

Return only valid JSON in the required schema.
""".strip()


CLEAN_RESPONSE_USER_PROMPT_TEMPLATE = """
Input:
{llm_response}

Output:

""".strip()


SIMPLEQA_CLEAN_RESPONSE_SYSTEM_PROMPT_TEMPLATE = """
Clean and normalize an LLM response to make it suitable for further factual verification.
This task contains two parts.

# First, remove low-value, non-informative text.
Remove expressions such as:
- Simple confirmations, such as "Yes," "Correct," or "Sure."
- Generic lead-ins, such as "The information is as follows" or "Here are the details."
- Empty transition phrases, such as "Specifically," "In particular," or "It should be noted that."
- Summary fillers, such as "In summary" or "Overall."
- duplicated content.
- Any other generic wording that does not add factual content.
Example:"Sure. Here are the details: Tesla reported revenue of $96.8 billion in 2023. Specifically, its automotive revenue was $82.4 billion.  " → "Tesla reported revenue of $96.8 billion in 2023. Its automotive revenue was $82.4 billion."
If the input contains incomplete fragments, labels, headings, or bullet-style fields, rewrite them as complete sentences when the missing sentence structure can be inferred from the local context.

# Second, normalize vague expressions.
- Rewrite vague qualifiers that express frequency, prevalence, tendency, or typicality into weaker possibility statements when their exact degree is not directly verifiable.
This includes words and phrases such as "often", "typically", "usually", "commonly", "sometimes", "frequently", "generally", "widely", "in many cases", "in some cases", and similar expressions.
Example: "The meat is typically beef." → "The meat can be beef."
- Rewrite example lists introduced by "such as", "including", or "for example" as explicit possible-value statements.
Example: "local spices such as cloves, and black pepper" → "The local spices can include cloves, and black pepper."
- Remove vague speculative claims.
Example: ”the Qurch often incorporates local spices from the broader biriyani spice palette,  which might be more pronounced than in some Gulf versions. "→"The Qurch can incorporate local spices from the broader biriyani spice palette."

Output format:
- Return only valid JSON.
- Do not add explanations, markdown, comments, or extra fields.  Return the result using exactly this schema:  {  "normalized_response": "cleaned and normalized response text" }
""".strip()

SIMPLEQA_CLEAN_RESPONSE_USER_PROMPT_TEMPLATE = """
Format example:

Input:
The Qurch often incorporates local spices from the broader biriyani spice palette, which might be more pronounced than in some Gulf versions.

Output:
{{
  "normalized_response": "The Qurch can incorporate local spices from the broader biriyani spice palette."
}}

Now process the following input and return only valid JSON.

Input:
{llm_response}

Output:
""".strip()

SIMPLEQA_RUBRICS_WRONG_SYSTEM_PROMPT = """
Rewrite the given rubrics into shorter rubrics.
Focus on identifying the incorrect or unsupported parts of the original content.
You can briefly mention the correct correction in parentheses, but do not make the correction the main focus.

Output format:
- One rubric per line.
- Do not use numbering, bullet points, or prefixes.
- Output only the rubrics, separated by newline characters.

Examples:
whether Richard Kestian declared "Thou art a lying knave," cannot be supported.
The accusation was not in Mr. Wymondsell or Dawes Wymondsell's house (actually in Sir Abraham Dawes's house).
""".strip()
SIMPLEQA_RUBRICS_WRONG_USER_PROMPT = """
Input:
{rubrics_wrong_orginal}
Output:
""".strip()


SPLIT_CLAIMS_SYSTEM_PROMPT_TEMPLATE = """
# Task
Extract explicit factual claims from the given text, splitting compound statements into standalone, complete, unambiguous claims that can be independently verified.

# Instructions

1. **One fact per predicate, explicit or implicit** 
  - Each extracted sentence covers exactly one predicate. 
  - Treat verifiable modifiers, appositives, titles, roles, and affiliations as implicit predicates when they can be rewritten as standalone facts. 
  - If multiple items share the same predicate(e.g., "X are covered by A, B and C together"), keep them together as one sentence.

2. **Make all subjects explicit and self-contained**
  - **Standalone**: Each fact must be understandable without reading other facts.
  - **No vague references**: Replace words like "it", "this", "the former", "that" with the full entity name.
  - **Repeat full identifier**: If an entity has a specific identifier (e.g., route number, official name), include it in EVERY fact that mentions it. Do NOT write it only once and then use a generic term later.


3. **Keep original phrasing** – Only change text to resolve references or disambiguate entities. Otherwise, keep the original wording.

4. Verifiable only – Include only explicit, checkable factual information. 
- Separate unrelated facts into different claims so each claim can be verified independently. 
- Exclude subjective opinions, emotional reactions, rhetorical exaggerations, and metaphorical descriptions.


# Output Format
* Output only the extracted facts, one per line. 
* Do not include any explanations, labels, or numbering.


""".strip()

SPLIT_CLAIMS_USER_PROMPT_TEMPLATE = """
Below are examples of decomposing a text into atomic, standalone facts.


# Examples

## Example 1
text: NVIDIA Corporation was founded by Taiwanese American electrical engineer Jensen Huang, Chris Malachowsky, and Curtis Priem in April 1993. It initially focused on graphics processing units. It invented the GPU in 1999, greatly advancing the development of computer graphics.
Output:
NVIDIA Corporation was founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in April 1993.
Jensen Huang is a Taiwanese American electrical engineer.
NVIDIA Corporation initially focused on graphics processing units.
NVIDIA Corporation invented the GPU in 1999.
NVIDIA Corporation’s invention of the GPU advanced the development of computer graphics.

## Example 2
text: Zootopia, produced by Walt Disney Pictures, was released in 2016 and depicts a modern city where predators and prey live together in harmony. The protagonist Judy is a small rabbit police officer who actively patrols the city and investigates cases. Her persistent efforts and energetic personality make her deeply inspiring, and every worker with dreams would be moved to tears when seeing her. The film eventually won the Best Animated Feature award at the 89th Academy Awards.
Output:
Zootopia was produced by Walt Disney Pictures.
Zootopia was released in 2016.
Zootopia depicts a modern city where predators and prey live together in harmony.
In Zootopia, Judy is a small rabbit police officer.
In Zootopia, Judy patrols the city and investigates cases.
Zootopia won the Best Animated Feature award at the 89th Academy Awards.



## Example 3
text: California's diverse landscape is anchored by the Sierra Nevada mountains, which house Mount Whitney. The state is also home to Redwood National Park. While the Mojave Desert contains Death Valley, the hottest place on Earth, the shimmering heat waves there seem to dance with a cruel, poetic beauty. 
Output:
California's landscape is anchored by the Sierra Nevada mountains.
The Sierra Nevada mountains house Mount Whitney.
The state California is also home to Redwood National Park.
The Mojave Desert in California contains Death Valley.
Death Valley is the hottest place on Earth.

# Task
Break down every sentence in the given text as required:
text: {}
Output:

""".strip()

Group_claims_SYSTEM_PROMPT = """
Put all atomic claims extracted from the same source text into a single question-level group.

Rules:
- Keep every claim exactly once.
- Do not omit, duplicate, rewrite, or translate any claim.
- Return only valid JSON that follows the schema shown in the example.
- Do not include explanations, markdown, comments, or extra fields.

Output schema:
{
  "group_name": "short descriptive group name for this question",
  "group_claims": [
    "claim1",
    "claim2"
  ]
}
""".strip()

GROUP_CLAIMS_USER_PROMPT_TEMPLATE = """
Input claims:
{claims_block}

Return only one group for this question as valid JSON:
{
  "group_name": "short descriptive group name for this question",
  "group_claims": [
    "claim1",
    "claim2"
  ]
}

Requirements:
- Put all current claims into `group_claims`.
- Keep the original claim text unchanged.
""".strip()

# Unified input field names expected by prompt builders.
FIELD_MODEL_ANSWER = "model_answer"
FIELD_CLEANED_MODEL_ANSWER = "cleaned_model_answer"
FIELD_CLAIMS = "claims"


def build_clean_prompts(clean_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build clean-stage prompts.

    Required input field:
    - model_answer
    """
    model_answer = str(clean_input.get(FIELD_MODEL_ANSWER) or "").strip()
    user_prompt = CLEAN_RESPONSE_USER_PROMPT_TEMPLATE.format(llm_response=model_answer)
    return CLEAN_RESPONSE_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_simpleqa_clean_prompts(clean_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build clean-stage prompts using the SimpleQA prompt variant."""
    model_answer = str(clean_input.get(FIELD_MODEL_ANSWER) or "").strip()
    user_prompt = SIMPLEQA_CLEAN_RESPONSE_USER_PROMPT_TEMPLATE.format(llm_response=model_answer)
    return SIMPLEQA_CLEAN_RESPONSE_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_split_prompts(split_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build split-stage prompts.

    Required input field:
    - cleaned_model_answer
    """
    cleaned_model_answer = str(split_input.get(FIELD_CLEANED_MODEL_ANSWER) or "").strip()
    user_prompt = SPLIT_CLAIMS_USER_PROMPT_TEMPLATE.format(cleaned_model_answer)
    return SPLIT_CLAIMS_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_group_prompts(group_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build group-stage prompts.

    Required input fields:
    - claims: Sequence[str]
    """
    claims = group_input.get(FIELD_CLAIMS) or []
    claims_block = json.dumps(list(claims), ensure_ascii=False, indent=2)
    user_prompt = GROUP_CLAIMS_USER_PROMPT_TEMPLATE.replace("{claims_block}", claims_block)
    return Group_claims_SYSTEM_PROMPT, user_prompt


# Compatibility wrappers for second_try pipeline.
def build_clean_prompt(model_answer: str) -> tuple[str, str]:
    """Wrapper: keep second_try API while using canonical prompt content."""
    return build_clean_prompts({FIELD_MODEL_ANSWER: model_answer})


def build_split_prompt(cleaned_response: str) -> tuple[str, str]:
    """Wrapper: keep second_try API while using canonical prompt content."""
    return build_split_prompts({FIELD_CLEANED_MODEL_ANSWER: cleaned_response})


def build_group_prompt(claims: Sequence[str], prior_topic_map: Mapping[str, str]) -> tuple[str, str]:
    """Wrapper: keep second_try API while using canonical prompt content."""
    return build_group_prompts(
        {
            FIELD_CLAIMS: claims,
        }
    )



SNIPPET_FILTER_SYSTEM_PROMPT_TEMPLATE = """
Your task is to filter current_round_snippets based on the claim, topic, and non_evidence_context.

Inputs:
- claim: The atomic claim currently being verified.
- topic: The topic group that the claim belongs to.
- non_evidence_context: Context for disambiguation only. Its factual correctness is unknown.
- current_round_snippets: Search result snippets from the current retrieval round. Each snippet contains candidate_id, url, title, and content.

Outputs:
- evidence_pool_candidate_ids: Snippet IDs that match the same context as the non_evidence_context, are relevant to the topic, and are worth adding to the topic evidence pool.
- direct_relevant_candidate_ids: A subset of evidence_pool_candidate_ids that can directly support, refute, or qualify the current claim.

Filtering rules:
1. Retain only snippets that match the entity, event, work, location, or context of the non_evidence_context, claim, and topic. Discard the rest.
2. Add snippet to evidence_pool_candidate_ids if it: (a) shares the same object as the non_evidence_context and the claim (content/opinion needn't match), (b) is topic-relevant, (c) may help verify claims under this topic.
3. Add snippet to direct_relevant_candidate_ids if it is already in evidence_pool_candidate_ids and directly supports, refutes, or qualifies the current claim.


Do not rewrite, translate, summarize, or complete any snippet.
Do not invent candidate IDs.
Return only valid JSON. Do not include explanations, markdown, comments, or extra fields.

Output schema:
{
  "evidence_pool_candidate_ids": ["C1", "C3"],
  "direct_relevant_candidate_ids": ["C1"]
}
""".strip()


SNIPPET_FILTER_USER_PROMPT_TEMPLATE = """
Process the following input.

claim:
{claim}

topic:
{topic}

non_evidence_context:
{non_evidence_context}

current_round_snippets:
{current_round_snippets}
""".strip()



NEXT_SEARCH_OR_ANSWER_SYSTEM_PROMPT_TEMPLATE = """
You are provided with a claim, non_evidence_context, present_snippets, and previous_searched_queries. 

Your task is to decide whether present_snippets are sufficient to verify the claim, or whether another search query is needed. if you thimk you still need more evience, raise another search query. 

You are also provided with topic_guidance_for_search, which is generated from previously collected evidence and can be treated as reliable guidance. It contains core background or information that may support or refute the claim. Use it to guide your reasoning and decisions.

**CRITICAL: non_evidence_context is NOT evidence.**  
It only helps identify the correct entity. Do NOT cite it as a fact, do NOT say "according to non_evidence_context", and do NOT use it to justify any part of the claim. All factual support must come from present_snippets. 


## Step 1 — Decide if present_snippets are sufficient
Use your basic logic and common sense.
You may answer only if present_snippets and topic_guidance_for_search contain **direct evidence** that:
- Supports **all** parts of the claim → answer `{factual_label}`
- Contradicts even only one part of the claim → answer `{non_factual_label}`

**Don't count something as evidence if:**
- The snippet is missing the exact name, number, date, location, or role you need.
- The snippet says something broader or related, but not exactly what the claim says.
- You have to assume, infer, or fill in missing info to make it fit.

If you're not sure and think there is no enough evidence — or if the snippet only talks about a different attribute or a vaguely related topic — do NOT answer. Instead, generate a search query.

## Step 2 — Generate a Search Query If Needed
If Step 1 does not provide enough direct evidence to answer, identify the missing information based on present_snippets and topic_guidance_for_search, then create one keyword-style Google search query to find that missing detail.
Guidelines:
- Use the main entity plus the specific detail being verified.
- Check present_snippets, previous_searched_queries, and topic_guidance_for_search before generating a query.
- Do not combine multiple unrelated missing facts into one query.

Avoid repeated queries:
- Do not make only minor wording changes to a previous query.
- Bad: My Love album released 2002; My Love released 2002 release date; My Love release date 2002.
- If the same information is still needed, change the search strategy:
  - Broader: My Love album release year
  - More specific: My Love album Alexander singer release year 2002
Examples:
claim example1: Mutsumi Tamura is known for the role of Bojji in Ranking of Kings.
search query: Mutsumi Tamura role Bojji Ranking of Kings
claim example2: Mia Serafino's character Mia worked George Nakai.
search query: Beef Mia work George Nakai 


OUTPUT FORMAT:
Return only valid JSON. Do not include explanations, markdown, comments, or extra fields.
When returning an answer, use exactly this JSON schema:
{{
  "action": "answer",
  "final_answer": "{factual_label} or {non_factual_label}",
  "reason": "Snippet S1 states that XXX joined the xxx Club in 1982, which is clearly contradictory to the year 1985 in the claim.",
  "evidence_snippet_ids": ["snippet_id"]
}}

When returning a search query, use exactly this JSON schema:
{{
  "action": "search",
  "search_query": "concise keyword-style search query intended to find needed information"
}}
""".strip()


NEXT_SEARCH_OR_ANSWER_USER_PROMPT_TEMPLATE = """
claim:
{claim}

non_evidence_context:
{non_evidence_context}

topic_guidance_for_search:
{topic_guidance}


present_snippets_with_ids:
{present_snippets_with_ids}


previous_searched_queries:
{searched_queries}


{last_step_feedback_block}
""".strip()

# below are for pre fetch before dealing with llm responses.
# just based on the input question
TOPIC_GROUNDING_QUERY_PLAN_SYSTEM_PROMPT_TEMPLATE = """
Generate search queries for topic-level grounding retrieval.

Requirements:
- Return exactly:
  - wiki_queries: {wiki_query_count} items
  - web_queries: {web_query_count} items
- wiki_queries must include key entity names from the question and include one of:
  - wiki
  - wikipedia
  - site:wikipedia.org
- web_queries must NOT contain wiki/wikipedia/site:wikipedia.org.
- Use concise keyword-style queries similar to human Google searches.
- wiki_queries should be shorter, compact keyword phrases.
- web_queries should be longer and include higher-disambiguation details from the question.

Return only valid JSON:
{{
  "wiki_queries": [
    "short query 1 with wiki term",
    "short query 2 with wiki term",
    "short query 3 with wiki term"
  ],
  "web_queries": [
    "longer query 1 without any wiki term",
    "longer query 2 without any wiki term",
    "longer query 3 without any wiki term"
  ]
}}

Hard validation rules:
- Array lengths must be exact: wiki_queries={wiki_query_count}, web_queries={web_query_count}.
- Every wiki query must include at least one key entity from topic_query.
- Every web query must include at least one key entity from topic_query.
- Do not output extra fields, markdown, comments, or explanation text.
""".strip()


TOPIC_GROUNDING_QUERY_PLAN_USER_PROMPT_TEMPLATE = """
topic_query:
{query}
""".strip()


TOPIC_GROUNDING_CANDIDATE_SELECT_SYSTEM_PROMPT_TEMPLATE = """
You are selecting snippets that are most likely to contain answer evidence or highly useful background.

Rules:
- You MUST select only from provided candidate_id values.
- Do NOT generate new candidate IDs.
- Do NOT output URLs.
- Each candidate already includes exactly: candidate_id, url, title, content.
- Prefer candidates that are specific, entity-matched, and likely evidence-bearing.
- Return at most {max_selected_candidates} candidate IDs.

Return only valid JSON:
{{
  "selected_candidate_ids": ["candidate_id_1", "candidate_id_2"]
}}

""".strip()


TOPIC_GROUNDING_CANDIDATE_SELECT_USER_PROMPT_TEMPLATE = """
topic_query:
{query}

deduplicated_candidates:
{candidates}
""".strip()


TOPIC_GROUNDING_EVIDENCE_EXTRACT_SYSTEM_PROMPT_TEMPLATE = """
Extract evidence from fetched full-page texts for topic-level grounding, and produce topic brief in the same output.

Context:
- The program already performed snippet dedup and LLM candidate selection.
- The fetched pages are from URLs mapped by selected candidate_id.

Output requirements:
- Return both `evidence_items` and `topic_brief`.
- Each item must include:
  - url_id
  - relevant_text (a relatively long direct text segment from source page)
  - summary(a complete summary of the relevant_text)
- Do not fabricate facts.
- Keep relevant_text tied to source text (do not output summary-only evidence).
- One URL may yield multiple evidence items.
- `topic_brief` is retrieval guidance, not just direct evidence; 
- `topic_brief` should be vivid and concrete in writing.
- Each factual statement in topic_brief must cite evidence snippet IDs like [S1].
- Use only topic_query and extracted evidence_items.
- Do not introduce new entities/dates/works not present in topic_query or evidence_items.

Return only valid JSON:
{{
  "evidence_items": [
    {{
      "snippet_id": "S1",
      "url_id": "397_EXT_U1",
      "relevant_text": "...",
      "summary": "..."
    }}
  ],
  "topic_brief": "..."
}}
""".strip()


TOPIC_GROUNDING_EVIDENCE_EXTRACT_USER_PROMPT_TEMPLATE = """
topic_id:
{topic_id}

topic_query:
{query}

fetched_pages_with_url_id:
{pages}
""".strip()


# topic_brief is now generated together with evidence extraction in one prompt.




MUST_HAVE_ANSWER_SYSTEM_PROMPT_TEMPLATE = """
You are provided with a claim, non_evidence_context, and present_snippets.
You are also provided with topic_guidance_for_search, which is generated from previously collected evidence and can be treated as reliable guidance. It contains core background or information that may support or refute the claim. Use it to guide your reasoning and decisions.

You cannot perform any more searches. You must return a final answer ({factual_label} or {non_factual_label}) based only on present_snippets.

**CRITICAL**: non_evidence_context is NOT evidence. Use it only to identify the correct entity. Never use it to justify a fact. All factual support must come from present_snippets.

## Step — Decide the answer based on present_snippets
Use your basic logic and common sense.
You may only return `{factual_label}` if present_snippets contain **direct evidence** that supports **all** parts of the claim.
You must return `{non_factual_label}` if:
- present_snippets contain direct evidence that contradicts even only one part of the claim, OR
- present_snippets do NOT provide direct evidence to support all parts of the claim (i.e., evidence missing, vague, off‑topic, or only related information).

**Don't count something as evidence if:**
- The snippet is missing the exact name, number, date, location, or role you need.
- The snippet says something broader or related, but not exactly what the claim says.
- You have to assume, infer, or fill in missing info to make it fit.


## Output format
When returning `{factual_label}`, `evidence_snippet_ids` must contain the IDs of snippets that directly support your reason.
Return only valid JSON. Do not include explanations, markdown, comments, or extra fields.
{{
  "final_answer": "{factual_label} or {non_factual_label}",
  "reason": "short reason based only on present_snippets. Must quote exact snippet text and show all reasoning steps derived from snippets. If any part of the claim cannot be fully determined from snippets, explicitly state that gap.",
  "evidence_snippet_ids": ["snippet_id"]
}}

Example:
{{
  "final_answer": "{factual_label}",
  "reason": "Snippet S1 states that XXX joined the xxx Club in 1982, which is clearly contradictory to the year 1985 in the claim.",
  "evidence_snippet_ids": ["S1"]
}}
""".strip()


MUST_HAVE_ANSWER_USER_PROMPT_TEMPLATE = """
claim:
{claim}

non_evidence_context:
{non_evidence_context}

topic_guidance_for_search:
{topic_guidance}

present_snippets_with_ids:
{present_snippets_with_ids}
""".strip()

# Evidence retrieval field names used by prompt builders.
FIELD_CLAIM = "claim"
FIELD_TOPIC = "topic"
FIELD_DISAMBIGUATION_TEXT = "non_evidence_context"
# Backward-compat alias for older callers.
FIELD_BACKGROUND_TEXT = "background_text"
FIELD_CURRENT_ROUND_SNIPPETS = "current_round_snippets"
FIELD_SEARCHED_QUERIES = "searched_queries"
FIELD_PRESENT_SNIPPETS_WITH_IDS = "present_snippets_with_ids"
FIELD_LAST_ERROR_TYPE = "last_error_type"
FIELD_LAST_ERROR_MESSAGE = "last_error_message"
FIELD_FAILED_QUERY = "failed_query"
FIELD_RETRY_ATTEMPTS_FOR_THIS_ROUND = "retry_attempts_for_this_round"
FIELD_DO_NOT_REPEAT_QUERIES = "do_not_repeat_queries"

# Unified verdict labels for retrieval stage.
FACTUAL_LABEL = "SUPPORTED"
NON_FACTUAL_LABEL = "REFUTED"


def _get_disambiguation_text(payload: Mapping[str, Any]) -> str:
    """Read disambiguation text with backward compatibility for old key names."""
    v = payload.get(FIELD_DISAMBIGUATION_TEXT)
    if v is None or str(v).strip() == "":
        v = payload.get(FIELD_BACKGROUND_TEXT)
    return str(v or "").strip()


def build_snippet_filter_prompt(snippet_filter_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build snippet-filter prompts.

    Required input fields:
    - claim: str
    - topic: str
    - non_evidence_context: str
    - current_round_snippets: list[dict] with candidate_id/url/title/content
    """
    claim = str(snippet_filter_input.get(FIELD_CLAIM) or "").strip()
    topic = str(snippet_filter_input.get(FIELD_TOPIC) or "").strip()
    non_evidence_context = _get_disambiguation_text(snippet_filter_input)
    current_round_snippets = snippet_filter_input.get(FIELD_CURRENT_ROUND_SNIPPETS) or []

    snippets_json = json.dumps(list(current_round_snippets), ensure_ascii=False, indent=2)
    user_prompt = SNIPPET_FILTER_USER_PROMPT_TEMPLATE.format(
        claim=claim,
        topic=topic,
        non_evidence_context=non_evidence_context,
        current_round_snippets=snippets_json,
    )
    return SNIPPET_FILTER_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_next_search_or_answer_prompt(next_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build NEXT_SEARCH_OR_ANSWER prompts for iterative retrieval.

    Required input fields:
    - claim: str
    - non_evidence_context: str
    - searched_queries: list[str]
    - present_snippets_with_ids: list[dict] with snippet_id/url/title/content
    """
    claim = str(next_input.get(FIELD_CLAIM) or "").strip()
    non_evidence_context = _get_disambiguation_text(next_input)
    searched_queries = next_input.get(FIELD_SEARCHED_QUERIES) or []
    present_snippets_with_ids = next_input.get(FIELD_PRESENT_SNIPPETS_WITH_IDS) or []
    last_error_type = str(next_input.get(FIELD_LAST_ERROR_TYPE) or "").strip()
    last_error_message = str(next_input.get(FIELD_LAST_ERROR_MESSAGE) or "").strip()
    failed_query = str(next_input.get(FIELD_FAILED_QUERY) or "").strip()
    retry_attempts_for_this_round = int(next_input.get(FIELD_RETRY_ATTEMPTS_FOR_THIS_ROUND) or 0)
    do_not_repeat_queries = [str(x).strip() for x in (next_input.get(FIELD_DO_NOT_REPEAT_QUERIES) or []) if str(x).strip()]
    topic_guidance = str(next_input.get("topic_guidance") or "").strip()

    last_step_feedback_block = ""
    if last_error_type or last_error_message or failed_query:
        feedback_obj = {
            "had_error": True,
            "error_type": last_error_type,
            "error_message": last_error_message,
            "failed_query": failed_query,
            "retry_attempts_for_this_round": max(0, retry_attempts_for_this_round),
            "do_not_repeat_queries": list(do_not_repeat_queries),
        }
        retry_instruction = (
            "If last_step_feedback is provided, you must read it first and avoid repeating failed queries. "
            "Then retry with a revised query or answer if evidence is sufficient."
        )
        last_step_feedback_block = (
            "last_step_feedback:\n"
            + json.dumps(feedback_obj, ensure_ascii=False, indent=2)
            + "\n"
            + retry_instruction
        )

    system_prompt = NEXT_SEARCH_OR_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        factual_label=FACTUAL_LABEL,
        non_factual_label=NON_FACTUAL_LABEL,
    )
    user_prompt = NEXT_SEARCH_OR_ANSWER_USER_PROMPT_TEMPLATE.format(
        claim=claim,
        non_evidence_context=non_evidence_context,
        topic_guidance=topic_guidance,
        searched_queries=json.dumps(list(searched_queries), ensure_ascii=False, indent=2),
        last_step_feedback_block=last_step_feedback_block,
        present_snippets_with_ids=json.dumps(
            list(present_snippets_with_ids),
            ensure_ascii=False,
            indent=2,
        ),
    )
    return system_prompt, user_prompt


def build_topic_grounding_query_plan_prompt(payload: Mapping[str, Any]) -> tuple[str, str]:
    query = str(payload.get("query") or "").strip()
    wiki_query_count = int(payload.get("wiki_query_count") or 3)
    web_query_count = int(payload.get("web_query_count") or 3)
    system_prompt = TOPIC_GROUNDING_QUERY_PLAN_SYSTEM_PROMPT_TEMPLATE.format(
        wiki_query_count=wiki_query_count,
        web_query_count=web_query_count,
    )
    user_prompt = TOPIC_GROUNDING_QUERY_PLAN_USER_PROMPT_TEMPLATE.format(
        query=query,
    )
    return system_prompt, user_prompt


def build_topic_grounding_url_select_prompt(payload: Mapping[str, Any]) -> tuple[str, str]:
    query = str(payload.get("query") or "").strip()
    candidates = payload.get("candidates") or []
    max_selected_candidates = int(payload.get("max_selected_candidates") or 8)
    system_prompt = TOPIC_GROUNDING_CANDIDATE_SELECT_SYSTEM_PROMPT_TEMPLATE.format(
        max_selected_candidates=max_selected_candidates
    )
    user_prompt = TOPIC_GROUNDING_CANDIDATE_SELECT_USER_PROMPT_TEMPLATE.format(
        query=query,
        candidates=json.dumps(list(candidates), ensure_ascii=False, indent=2),
    )
    return system_prompt, user_prompt


def build_topic_grounding_evidence_extract_prompt(payload: Mapping[str, Any]) -> tuple[str, str]:
    topic_id = str(payload.get("topic_id") or "").strip()
    query = str(payload.get("query") or "").strip()
    pages = payload.get("pages") or []
    system_prompt = TOPIC_GROUNDING_EVIDENCE_EXTRACT_SYSTEM_PROMPT_TEMPLATE
    user_prompt = TOPIC_GROUNDING_EVIDENCE_EXTRACT_USER_PROMPT_TEMPLATE.format(
        topic_id=topic_id,
        query=query,
        pages=json.dumps(list(pages), ensure_ascii=False, indent=2),
    )
    return system_prompt, user_prompt


def build_topic_grounding_topic_brief_prompt(payload: Mapping[str, Any]) -> tuple[str, str]:
    raise NotImplementedError(
        "topic_brief prompt is merged into build_topic_grounding_evidence_extract_prompt; "
        "use that function and read `topic_brief` from the same JSON output."
    )


def build_must_have_answer_prompt(must_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build MUST_HAVE_ANSWER prompts when retrieval rounds are exhausted.

    Required input fields:
    - claim: str
    - non_evidence_context: str
    - present_snippets_with_ids: list[dict] with snippet_id/url/title/content
    """
    claim = str(must_input.get(FIELD_CLAIM) or "").strip()
    non_evidence_context = _get_disambiguation_text(must_input)
    topic_guidance = str(must_input.get("topic_guidance") or "").strip()
    present_snippets_with_ids = must_input.get(FIELD_PRESENT_SNIPPETS_WITH_IDS) or []

    system_prompt = MUST_HAVE_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        factual_label=FACTUAL_LABEL,
        non_factual_label=NON_FACTUAL_LABEL,
    )
    user_prompt = MUST_HAVE_ANSWER_USER_PROMPT_TEMPLATE.format(
        claim=claim,
        non_evidence_context=non_evidence_context,
        topic_guidance=topic_guidance,
        present_snippets_with_ids=json.dumps(
            list(present_snippets_with_ids),
            ensure_ascii=False,
            indent=2,
        ),
    )
    return system_prompt, user_prompt


RUBRICS_VERDICT_SYSTEM_PROMPT_TEMPLATE = """
You are evaluating the quality and coverage of verifier_output.

You are provided with:
- original_response: the response being checked. The claims in verifier_output were decomposed from this response.
- verifier_output: claim-level factual verification results for original_response. Each claim includes a verdict plus a rationale.
- rubrics_correct: a checklist of factual points extracted from original_response that human annotators consider correct. The verifier_output is expected to affirm, support, or preserve these facts.
- rubrics_wrong: a checklist of factual problems extracted from original_response that human annotators expect the verifier_output to catch. Some items in rubrics_wrong include the correct replacement for context.

Your task is to judge whether verifier_output as a whole successfully covered each rubric item.
When judging each rubric item:
- Use both the claim verdicts and, more importantly, the rationales in verifier_output.
- Aggregate all relevant claims and rationales. Do not assume that one rubric item maps to exactly one claim, or that one claim maps to exactly one rubric item.


For rubrics_correct:
- Look across all relevant claims and rationales in verifier_output, not just one claim.
- Mark "hit" if those claims and rationales recognize, support, or preserve the fact as correct.
- Mark "miss" if those claims and rationales ignore the fact, reject it, mark it unsupported, or treat it as incorrect.

For rubrics_wrong:
- Identify the erroneous or unsupported information described by the rubric item. You may IGNORE any correction when deciding hit/miss, because the verifier_output only needs to catch the error, not provide the correct replacement.
- Look across all relevant claims and rationales in verifier_output, not just one claim.
- Mark "hit" if verifier_output rejects, questions, refutes, or marks that wrong claim as unsupported.
- Mark "miss" only if verifier_output does not address the wrong claim, treats it as correct, or discusses only an unrelated issue.
- Never mark "miss" with this logic: "verifier_output refuted the wrong claim/capture the error, but did not mention the corrected replacement." That logic is invalid; it should be "hit" instead.



Return only valid JSON using this schema:
{
  "correct_claims": [
    {
      "rubric": "The runway at Fath Air Base was too short for a Boeing 707.",
      "result": "hit",
      "reason": "The verifier_output recognizes this fact as correct."
    }
  ],
  "wrong_claims": [
    {
      "rubric": "Edward A. Tuttle’s patent was No. 248,121, not No. 248,124.",
      "result": "hit",
      "reason": "The verifier_output identifies that the patent number in the response is unsupported."
    },
    {
      "rubric": "The claimed source was Court of Chivalry 1634–1640, not Records of the Court of the Star Chamber.",
      "result": "hit",
      "reason": "The verifier_output marks the Star Chamber source claim as unsupported by the evidence."
    },
    {
      "rubric": "The incident occurred at Sir Abraham Dawes’s house, not John Popham’s house.",
      "result": "hit",
      "reason": "The verifier_output rejects that the incident occurred at John Popham's house,saying it is not fully supported by evidence."
    }
  ]
}
- Return one output item for every rubric item.
- If rubrics_correct is empty, return "correct_claims": [].
- If rubrics_wrong is empty, return "wrong_claims": [].
- Do not invent new rubric items.


""".strip()


RUBRICS_VERDICT_USER_PROMPT_TEMPLATE = """
Please process the following inputs.
verifier_output:
{verdict_of_each_claim}

original_response:
{original_response}

rubrics_correct:
{rubrics_correct}

rubrics_wrong:
{rubrics_wrong}


""".strip()


FIELD_VERDICT_OF_EACH_CLAIM = "verdict_of_each_claim"
FIELD_ORIGINAL_RESPONSE = "original_response"
FIELD_RUBRICS_CORRECT = "rubrics_correct"
FIELD_RUBRICS_WRONG = "rubrics_wrong"


def _normalize_rubric_points(rubrics_value: Any) -> list[str]:
    """Normalize rubrics input into a list of non-empty rubric points."""
    if isinstance(rubrics_value, list):
        return [str(x).strip() for x in rubrics_value if str(x).strip()]
    text = str(rubrics_value or "").strip()
    if not text:
        return []
    if text in {"no_correct_points", "no_wrong_points", "no_correct", "no_wrong"}:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _normalize_verdict_items(verdict_items: Any) -> list[dict[str, str]]:
    """Normalize claim verdict rows for rubrics-verdict prompt."""
    out: list[dict[str, str]] = []
    for row in (verdict_items or []):
        if not isinstance(row, Mapping):
            continue
        claim = str(row.get("claim") or row.get("atomic_claim") or "").strip()
        verdict = str(
            row.get("final_answer")
            or row.get("auto_verdict")
            or row.get("verdict")
            or ""
        ).strip()
        reason = str(
            row.get("reason")
            or row.get("verdict_rationale")
            or row.get("rationale")
            or ""
        ).strip()
        if not claim:
            continue
        out.append(
            {
                "claim": claim,
                "verdict": verdict,
                # Backward-compatible alias for existing readers.
                "final_answer": verdict,
                "reason": reason,
            }
        )
    return out


def build_rubrics_verdict_prompt(rubrics_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build RUBRICS_VERDICT prompts.

    Required input fields:
    - verdict_of_each_claim: list[dict], each item includes claim + verdict label + reason
    - original_response: str
    - rubrics_correct: str | list[str]
    - rubrics_wrong: str | list[str]
    """
    verdict_items = _normalize_verdict_items(rubrics_input.get(FIELD_VERDICT_OF_EACH_CLAIM) or [])
    original_response = str(rubrics_input.get(FIELD_ORIGINAL_RESPONSE) or "").strip()
    rubrics_correct = _normalize_rubric_points(rubrics_input.get(FIELD_RUBRICS_CORRECT))
    rubrics_wrong = _normalize_rubric_points(rubrics_input.get(FIELD_RUBRICS_WRONG))

    user_prompt = RUBRICS_VERDICT_USER_PROMPT_TEMPLATE.format(
        verdict_of_each_claim=json.dumps(verdict_items, ensure_ascii=False, indent=2),
        original_response=original_response,
        rubrics_correct=json.dumps(rubrics_correct, ensure_ascii=False, indent=2),
        rubrics_wrong=json.dumps(rubrics_wrong, ensure_ascii=False, indent=2),
    )
    return RUBRICS_VERDICT_SYSTEM_PROMPT_TEMPLATE, user_prompt
