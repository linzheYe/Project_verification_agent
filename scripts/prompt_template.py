from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
# claim prepare: clean split 
# prefetch
# retrive
# rubric
CLEAN_RESPONSE_SYSTEM_PROMPT_TEMPLATE = """
Clean an LLM response. Keep facts and remove conversational noise. This is a cleanup task, not a rewriting task.

Follow these steps:

1. Keep factual information.
- Keep every factual statement from the input.
- Keep names, dates, numbers, titles, and technical terms as written.
- When a noisy sentence contains a fact, keep the fact and remove the noise.

2. Remove noise.
- Remove introductory phrases that do not add information.
- Remove text saying the model is unsure, does not know, cannot answer, or does not want to guess.
- Remove text telling the user to check other sources.
- Remove questions to the user.
- Remove text about what the user may know, think, mean, find, or want.
  Examples: "You may be thinking of...", "Maybe you mean...", "You may want to know...", "If you read ..., you will find ...".


3. Adjust wording only when needed.
- Keep the original wording when it is already clear.
- Prefer deletion over rewriting.
- Rewrite only when labels, fields, fragments, or bullet points need to become a complete sentence.
- Rewrite only when removing text would leave an incomplete sentence.


Do not:
- Add facts.
- Summarize.
- Rewrite facts unless necessary.
- Make uncertain wording sound more certain.
- Remove factual information because it seems repetitive.

If no factual information remains, return an empty string.


Return only valid JSON using exactly this schema:
{
  "cleaned_response": "cleaned response text"
}

Example:

Input:
Selection batch: Third batch
Selection date: June 8, 2011

Output:
{
  "cleaned_response": "The item was selected in the third batch, with a selection date of June 8, 2011."
}

Input:
I don't have confident information about the specific city where the conference started. I'd recommend checking official sources.

Output:
{
  "cleaned_response": ""
}

Input:
I'm not confident enough to name a specific EP. In 2019, Rosalía was primarily releasing singles like "Con Altura" (with J Balvin) and "Aute Cuture."

Output:
{
  "cleaned_response": "In 2019, Rosalía was primarily releasing singles like \\"Con Altura\\" (with J Balvin) and \\"Aute Cuture.\\""
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


#- topic: Optional. The topic group that the claim belongs to.
SNIPPET_FILTER_SYSTEM_PROMPT_TEMPLATE = """
Your task is to filter current_round_snippets based on the inputs.


Inputs:
- claim: The atomic claim currently being verified.
- response_context: The original response that contains TARGET_CLAIM. Use it only to understand what the target claim refers to.
- current_round_snippets: search result snippets from the latest retrieval round. Each snippet contains candidate_id, url, title, and content.
- existing_claim_snippets: snippets already collected for verifying this claim. Use them only to judge whether a current-round snippet adds new information.



Outputs:
- evidence_pool_candidate_ids: candidate_ids that can help verify (support or refute) this claim or other claims from response_context.
- direct_relevant_candidate_ids: A subset of evidence_pool_candidate_ids that directly support, refute, or qualify the claim.


Filtering rules:
For each snippet in current_round_snippets:

1. First decide whether it adds useful evidence.
   Add it to evidence_pool_candidate_ids only if both conditions are met:
   - It provides factual information that may help verify the claim or closely related claims in response_context.
   - It adds new information not already covered by existing_claim_snippets, or provides a clearly more direct, reliable, or claim-specific formulation of the same fact.

2. Then decide whether it is directly relevant to the claim.
   Add it to direct_relevant_candidate_ids only if it directly supports, refutes, or qualifies the claim itself.

Constraint:
direct_relevant_candidate_ids must be a subset of evidence_pool_candidate_ids.


Do not rewrite, summarize, or complete any snippet.
Do not invent candidate IDs.
Return only valid JSON. Do not include explanations, markdown, comments, or extra fields.

Output schema:
{
  "evidence_pool_candidate_ids": ["C1", "C3"],
  "direct_relevant_candidate_ids": ["C1"]
}
""".strip()

SNIPPET_FILTER_USER_PROMPT_TEMPLATE = """
Process the following input. Select evidence_pool_candidate_ids from current_round_snippets, then select direct_relevant_candidate_ids as its subset, using existing_claim_snippets to avoid redundancy.

claim:
{claim}

response_context:
{response_context}

current_round_snippets:
{current_round_snippets}

existing_claim_snippets:
{existing_claim_snippets}

""".strip()



NEXT_SEARCH_OR_ANSWER_SYSTEM_PROMPT_TEMPLATE = """
You are provided with:
- TARGET_CLAIM: the claim that needs to be checked.
- response_context: the original response containing TARGET_CLAIM. Use it ONLY to understand what TARGET_CLAIM refers to.
- search_history: retrieval history by round. Each item contains the round_index, the search query used in that round, the newly retrived direct-evidence snippet ids and content using this search query.
- topic_guidance_for_search: guidance for understanding the topic.


**Task**:
Your task is to decide whether you should answer or return a search query. Answer only if the evidence in search_history directly supports all factual elements in the TARGET_CLAIM or directly contradicts at least one; if anything is missing or uncertain in the TARGET_CLAIM, generate another search query.



## Step 1 — Decide if search_history is sufficient
Before deciding, identify ALL factual elements in TARGET_CLAIM: named person, place, source, date, quote, role, relation, and value. You have to decide whether the evidence support TARGET_CLAIM exactly as written.

### Decision rules
Use the snippets inside search_history as factual evidence. Compare the claim against the evidence element by element, and keep this question in mind: does the evidence state the same thing, state a different thing, or fail to state it? Also use search_history to understand which previous search queries already produced useful direct evidence and which did not, so your next query targets a missing factual element instead of repeating a failed direction.
- Full direct support for all factual elements → `{factual_label}`
- Direct contradiction of any factual element → `{non_factual_label}`
- Incomplete, vague, indirect, or uncertain evidence without direct contradiction → generate a search query

Partial support is not enough for `{factual_label}`. A different value for the same exact entity and relation is a direct contradiction.

**IMPORTANT**:Be highly cautious with `{factual_label}`. Re-check it as if it may be wrong: every factual element must be directly supported, without any missing or contradictory part.




### Common errors to avoid in Step 1
**Example 1: matched description, wrong object**
TARGET_CLAIM: The crest depicts a wolf rampant argent.
Snippet evidence: "The crest depicts a stag rampant argent."
Element check: the description "rampant argent" matches, but the object does not. The claim requires a wolf, while the snippet states a stag.
Decision: `{non_factual_label}`.

**Example 2: event supported, named person missing**
TARGET_CLAIM: The dispute between Henry Cole and Martin Avery occurred in front of Sir William Harcourt.
Snippet evidence: "Henry Cole accused Martin Avery during a hearing in the presence of local magistrates."
Element check: the dispute and setting are related, but the named person is missing. The claim requires Sir William Harcourt, while the snippet only states local magistrates.
Decision: generate a search query.

**Example 3: date supported, named source contradicted**  
TARGET_CLAIM: The club notice in The London Gazette is dated April 1824.  
Snippet evidence: "The club notice was published in The Manchester Courier on 14 April 1824."  
Element check: the date April 1824 is supported, but the named source is contradicted. The claim requires The London Gazette, while the snippet states The Manchester Courier.  
Decision: `{non_factual_label}`.



## Step 2 — Generate a Search Query If Needed
If Step 1 does not provide enough direct evidence to verify ALL factual elements in the TARGET_CLAIM, identify the missing factual element and create one keyword-style Google search query for it. Use the fewest words when generating the search query.
Review each prior search query and its returned snippets. If they still leave the missing fact unresolved, treat the query direction as wrong and issue a new query targeting a different angle.

**Query rules:**
- Use core keywords only: names, nouns, domain terms, and target values. Remove extra context and low-value verbs such as “means”, “signifies”, “represents”, “occurred”, or “refers to”.
- Search only the missing factual element, not the whole TARGET_CLAIM. Do not add unnecessary known details just because they appear in the claim.
- Do not mix separate missing facts in one query.
- If a similar query was already tried, shorten it to core names or terms instead of rephrasing it.


**Examples:**
TARGET_CLAIM: Mutsumi Tamura is known for the role of Bojji in Ranking of Kings.
search query: Mutsumi Tamura Bojji Ranking of Kings

TARGET_CLAIM: Mia Serafino's character Mia worked for George Nakai in TV series Beef.
search query: Mia George Nakai Beef  

TARGET_CLAIM: The family crest depicts a falcon displayed Proper, meaning the falcon is shown in its natural colors.
search query: heraldry Proper natural colors

TARGET_CLAIM: House Loglas has a wolf argent on its shield, in heraldry meaning the wolf is silver.
Do not search “argent silver wolf shield.” Search one fact at a time, do not mix separate missing facts in one query: “heraldry argent silver” or “House Loglas shield wolf.”

TARGET_CLAIM: In medical terminology, "chronic cough in children" means the cough lasts a long time.
Do not search `medical terminology chronic cough in children lasts long time`.
Search `medical terminology chronic long lasting`; the intended missing fact is the meaning of "chronic"; "children" is irrelevant to it.


## OUTPUT FORMAT:
Return only valid JSON. Do not include explanations, markdown, comments, or extra fields.

- **When returning an answer with `{non_factual_label}` label, use exactly this JSON schema:**
{{
  "action": "answer",
  "reasoning": "TARGET_CLAIM requires: person Arthur Bennett, club Riverside Cricket Club, relation joined, date 1985, source The London Gazette. Snippets S1 and S2 state: Arthur Bennett joined Riverside Cricket Club in 1985, and the record was published in The Manchester Courier. The person, club, year and relation match, but the source is The Manchester Courier, not The London Gazette. The named source is directly contradicted, so the answer is {non_factual_label}.",
  "evidence_snippet_ids": ["S1", "S2"],
  "final_answer": "{non_factual_label}"
}}

- **When returning an answer with `{factual_label}` label, use exactly this JSON schema:**
{{
  "action": "answer",
  "reasoning": "TARGET_CLAIM requires: person XXX, club xxx Club, relation joined, date 1982. Snippet S1 states the same: XXX joined the xxx Club in 1982. The person, club, relation, and date all match exactly, so the answer is {factual_label}.",
  "evidence_snippet_ids": ["S1"],
  "final_answer": "{factual_label}"
}}


- **When returning a search query, use exactly this JSON schema:**
{{
  "action": "search",
  "search_query": "Use the fewest words needed for retrieval"
}}
""".strip()

NEXT_SEARCH_OR_ANSWER_USER_PROMPT_TEMPLATE = """
TASK:
Decide whether to answer now or generate another search query.
Answer only on full direct support or direct contradiction. otherwise search.

IMPORTANT:
Do not give {factual_label} lightly. Start by assuming TARGET_CLAIM is wrong. Then demand direct evidence for TARGET_CLAIM. In your reasoning, explicitly check every part of TARGET_CLAIM against the evidence.


Inputs:
TARGET_CLAIM:
{claim}

response_context:
{response_context}

topic_guidance_for_search:
{topic_guidance}

search_history:
{search_history}

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


TOPIC_GROUNDING_EVIDENCE_EXTRACT_WITH_ANSWER_SYSTEM_PROMPT_TEMPLATE = """
Extract evidence from fetched full-page texts for topic-level grounding, and produce topic brief in the same output.

Context:
- The program already performed snippet dedup and LLM candidate selection.
- The fetched pages are from URLs mapped by selected candidate_id.
- In this variant, a proposed answer to topic_query is also provided.

Output requirements:
- Return both `evidence_items` and `topic_brief`.
- Also return `answer_assessment`.
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

Answer assessment requirements:
- `answer_assessment` evaluates whether the provided_answer is supported as an answer to topic_query.
- Base the judgment on fetched evidence and extracted evidence_items. Use topic_brief only as a concise synthesis of the same evidence, not as an independent source.
- `answer_assessment.reasoning` must cite evidence snippet IDs like [S1] or [S1][S2], in the same style as topic_brief.
- `answer_assessment.reasoning` must reason step by step. Keep the wording simple and direct.
- In the JSON object for `answer_assessment`, output `reasoning` before the boolean judgment fields.
- Do not output a separate `supporting_snippet_ids` field.
- Use this decision process:
  1. State what field the question is asking for.
  2. State what each useful snippet says for that field.
  3. Decide whether the snippets agree, conflict, or do not directly answer that field.
  4. Then decide `is_supported`, `is_unique_answer`, and `canonical_answer`.
- Set `is_supported` to true only when the evidence set supports the provided_answer and no snippet gives a different value for the same field.
- Set `is_unique_answer` to true only when the evidence set gives one answer for the asked field and no snippet gives a different value for that same field.
- If one direct table, infobox, or award/filmography row gives the answer and the other snippets do not conflict with it, then `is_unique_answer` should usually be true.
- If some snippet is broader, uses a different label, or talks about a related field, do not treat it as direct conflict unless it gives a different value for the same field asked by the question.
- If the evidence does not clearly resolve one corrected answer, leave `canonical_answer` as an empty string.

Few-shot example 1:
Query: On what date was the soundtrack released?
Provided answer: August 10, 1995
Snippet S1: The soundtrack was released on August 10, 1995. [S1]
Snippet S2: The soundtrack was released on July 11, 1995. [S2]
Reasoning:
Step 1: The question asks for the soundtrack release date.
Step 2: S1 gives August 10, 1995 for that date [S1].
Step 3: S2 gives July 11, 1995 for that same date [S2].
Step 4: These two snippets give different values for the same field [S1][S2].
Step 5: The evidence conflicts, so the provided answer is not supported and the answer is not unique [S1][S2].
Output:
{{
  "answer_assessment": {{
    "provided_answer": "August 10, 1995",
    "reasoning": "Step 1: The question asks for the soundtrack release date. Step 2: S1 gives August 10, 1995 for that date [S1]. Step 3: S2 gives July 11, 1995 for that same date [S2]. Step 4: These two snippets give different values for the same field [S1][S2]. Step 5: The evidence conflicts, so the provided answer is not supported and the answer is not unique [S1][S2].",
    "is_supported": false,
    "is_unique_answer": false,
    "canonical_answer": ""
  }}
}}

Few-shot example 2:
Query: How many films did the actor appear in during 2022?
Provided answer: 3
Snippet S1: A filmography table lists three 2022 film rows. [S1]
Snippet S2: One of those films is confirmed as a 2022 title. [S2]
Snippet S3: Another of those films is confirmed as a 2022 title. [S3]
Reasoning:
Step 1: The question asks for the number of 2022 films.
Step 2: S1 directly gives three film rows for 2022 [S1].
Step 3: S2 and S3 support that those rows are real 2022 films [S2][S3].
Step 4: No snippet gives a different count for 2022 [S1][S2][S3].
Step 5: The evidence supports 3, and the answer is unique in this evidence set [S1][S2][S3].
Output:
{{
  "answer_assessment": {{
    "provided_answer": "3",
    "reasoning": "Step 1: The question asks for the number of 2022 films. Step 2: S1 directly gives three film rows for 2022 [S1]. Step 3: S2 and S3 support that those rows are real 2022 films [S2][S3]. Step 4: No snippet gives a different count for 2022 [S1][S2][S3]. Step 5: The evidence supports 3, and the answer is unique in this evidence set [S1][S2][S3].",
    "is_supported": true,
    "is_unique_answer": true,
    "canonical_answer": "3"
  }}
}}

Few-shot example 3:
Query: On what date did the person begin the office of Head of State?
Provided answer: January 1, 1925
Snippet S1: A table labeled "Head of State" gives term start as January 1, 1925. [S1]
Snippet S2: A broader list of heads of state gives a longer tenure starting June 28, 1922. [S2]
Snippet S3: A president table gives a term start of June 29, 1922. [S3]
Reasoning:
Step 1: The question asks for the start date of the office "Head of State".
Step 2: S1 directly gives that field: January 1, 1925 [S1].
Step 3: S2 and S3 use broader or different office labels [S2][S3].
Step 4: Because S2 and S3 do not give the same field in the same way, they are not direct conflict with S1 for this question [S1][S2][S3].
Step 5: The evidence supports January 1, 1925, and that answer is unique in this evidence set [S1].
Output:
{{
  "answer_assessment": {{
    "provided_answer": "January 1, 1925",
    "reasoning": "Step 1: The question asks for the start date of the office 'Head of State'. Step 2: S1 directly gives that field: January 1, 1925 [S1]. Step 3: S2 and S3 use broader or different office labels [S2][S3]. Step 4: Because S2 and S3 do not give the same field in the same way, they are not direct conflict with S1 for this question [S1][S2][S3]. Step 5: The evidence supports January 1, 1925, and that answer is unique in this evidence set [S1].",
    "is_supported": true,
    "is_unique_answer": true,
    "canonical_answer": "January 1, 1925"
  }}
}}

Few-shot example 4:
Query: For how many years was the region administered as a fief during 1703–1784?
Provided answer: 78
Snippet S1: A table shows fief from 1703 to 1745, then not-fief from 1745 to 1748, then fief again from 1748 to 1784. [S1]
Snippet S2: A second source repeats the same two fief intervals: 1703–1745 and 1748–1784. [S2]
Reasoning:
Step 1: The question asks for the total number of years in the fief periods within 1703–1784.
Step 2: S1 gives two fief intervals: 1703–1745 and 1748–1784, with a non-fief gap in between [S1].
Step 3: S2 gives the same two fief intervals [S2].
Step 4: Using the same year-boundary convention shown by the table, the durations are (1745 - 1703) and (1784 - 1748), which sum to 78 [S1][S2].
Step 5: No snippet gives different interval boundaries or a different total for this same field [S1][S2].
Step 6: The evidence supports 78, and the answer is unique in this evidence set [S1][S2].
Output:
{{
  "answer_assessment": {{
    "provided_answer": "78",
    "reasoning": "Step 1: The question asks for the total number of years in the fief periods within 1703–1784. Step 2: S1 gives two fief intervals: 1703–1745 and 1748–1784, with a non-fief gap in between [S1]. Step 3: S2 gives the same two fief intervals [S2]. Step 4: Using the same year-boundary convention shown by the table, the durations are (1745 - 1703) and (1784 - 1748), which sum to 78 [S1][S2]. Step 5: No snippet gives different interval boundaries or a different total for this same field [S1][S2]. Step 6: The evidence supports 78, and the answer is unique in this evidence set [S1][S2].",
    "is_supported": true,
    "is_unique_answer": true,
    "canonical_answer": "78"
  }}
}}

Few-shot example 5:
Query: What is the rank number of Company X among the largest private employers in the state as of March 2019?
Provided answer: 5
Snippet S1: A source says Company X is in the top five employers. [S1]
Snippet S2: Another source gives a March 2014 top-five list and includes Company X, but does not say its exact March 2019 rank. [S2]
Reasoning:
Step 1: The question asks for one exact field: the rank number of Company X as of March 2019.
Step 2: S1 only says Company X is in the top five, not that it is ranked 5th [S1].
Step 3: S2 also fails to give the exact March 2019 rank number, and it is about a different year [S2].
Step 4: The evidence does not give the exact requested value for the asked field [S1][S2].
Step 5: Because the exact rank number is missing, the provided answer is not supported, and the answer is not unique [S1][S2].
Output:
{{
  "answer_assessment": {{
    "provided_answer": "5",
    "reasoning": "Step 1: The question asks for one exact field: the rank number of Company X as of March 2019. Step 2: S1 only says Company X is in the top five, not that it is ranked 5th [S1]. Step 3: S2 also fails to give the exact March 2019 rank number, and it is about a different year [S2]. Step 4: The evidence does not give the exact requested value for the asked field [S1][S2]. Step 5: Because the exact rank number is missing, the provided answer is not supported, and the answer is not unique [S1][S2].",
    "is_supported": false,
    "is_unique_answer": false,
    "canonical_answer": ""
  }}
}}

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
  "topic_brief": "...",
  "answer_assessment": {{
    "provided_answer": "...",
    "reasoning": "Step 1: ... [S1] Step 2: ... [S2] Step 3: ... Therefore ... [S1][S2]",
    "is_supported": true,
    "is_unique_answer": false,
    "canonical_answer": "..."
  }}
}}
""".strip()


TOPIC_GROUNDING_EVIDENCE_EXTRACT_WITH_ANSWER_USER_PROMPT_TEMPLATE = """
topic_id:
{topic_id}

topic_query:
{query}

provided_answer:
{answer}

fetched_pages_with_url_id:
{pages}
""".strip()


# topic_brief is now generated together with evidence extraction in one prompt.




MUST_HAVE_ANSWER_SYSTEM_PROMPT_TEMPLATE = """
You are provided with:
- TARGET_CLAIM: the claim that needs to be checked.
- response_context: the original response containing TARGET_CLAIM. Use it ONLY to understand what TARGET_CLAIM refers to.
- present_snippets: the evidence collected so far. Use it to decide whether TARGET_CLAIM is supported or contradicted.
- topic_guidance_for_search: guidance for understanding the topic.

## Task
Based on the present_snippets, you should return a final answer: `{factual_label}`, `{non_factual_label}`, or `{nei_label}`.


## Decision rules
Before deciding, identify all factual elements in the TARGET_CLAIM.

Return `{factual_label}` only if present_snippets directly support every factual elements in the TARGET_CLAIM.

Return `{non_factual_label}` if:
- present_snippets directly contradict at least one factual elements in TARGET_CLAIM. 
- present_snippets have reached the right area of evidence for TARGET_CLAIM, but at least one detail in the claim is still not directly supported. This includes cases where support is missing, vague, only indirectly related, or requires inference.

Return `{nei_label}` only if present_snippets have not reached the right area of evidence, such as when they discuss another entity, another event, another term, or another relation. 


## Common error to avoid:
TARGET_CLAIM: The crest depicts a wolf rampant argent.
Snippet evidence: "The crest depicts a stag rampant argent."
Do NOT answer {factual_label} by saying the snippet matches "rampant argent." The required animal is different: wolf ≠ stag. This is {non_factual_label}.


## Output format
When returning `{factual_label}`, `evidence_snippet_ids` must contain the IDs of snippets that directly support your reason.
Return only valid JSON. Do not include explanations, markdown, comments, or extra fields.
{{
  "reasoning": "short reason based only on present_snippets. Must quote exact snippet text and show all reasoning steps derived from snippets. If any part of the TARGET_CLAIM cannot be fully determined from snippets, explicitly state that gap.",
  "evidence_snippet_ids": ["snippet_id"],
  "final_answer": "{factual_label} or {non_factual_label} or {nei_label}"
}}

Examples:
{{
  "action": "answer",
  "reasoning": "TARGET_CLAIM requires: person XXX, club xxx Club, relation joined, date 1982. Snippet S1 states the same: XXX joined the xxx Club in 1982. The person, club, relation, and date all match exactly, so the answer is {factual_label}.",
  "evidence_snippet_ids": ["S1"],
  "final_answer": "{factual_label}"
}}

{{
  "action": "answer",
  "reasoning": "TARGET_CLAIM requires: person Arthur Bennett, club Riverside Cricket Club, relation joined, date 1985, source The London Gazette. Snippets S1 and S2 state: Arthur Bennett joined Riverside Cricket Club in 1985, and the record was published in The Manchester Courier. The person, club, year and relation match, but the source is The Manchester Courier, not The London Gazette. The named source is directly contradicted, so the answer is {non_factual_label}.",
  "evidence_snippet_ids": ["S1", "S2"],
  "final_answer": "{non_factual_label}"
}}


{{
  "action": "answer",
  "reasoning": "TARGET_CLAIM requires: painting The Blue Garden, relation exhibited at, source Royal Academy catalogue. Snippet S1 states: \"The Blue Garden was exhibited in London in 1904.\" The snippet has reached the right area of evidence and supports the painting and exhibition, but it does not state the Royal Academy catalogue as the source. The required source is missing, so the answer is {non_factual_label}.",
  "evidence_snippet_ids": ["S1"],
  "final_answer": "{non_factual_label}"
}}

{{
  "action": "answer",
  "reasoning": "TARGET_CLAIM requires: term adding machine, relation also called, alternate term calculating machine. Snippets S1 and S2 discuss an exercising machine patent, and Snippet S3 only defines calculating machine generally. None of the snippets mention adding machine or state that adding machine is also called calculating machine. The snippets have not reached the right area of evidence for TARGET_CLAIM, so the answer is {nei_label}.",
  "evidence_snippet_ids": ["S1", "S2", "S3"],
  "final_answer": "{nei_label}"
}}

""".strip()


MUST_HAVE_ANSWER_USER_PROMPT_TEMPLATE = """
TARGET_CLAIM:
{claim}

response_context:
{response_context}

topic_guidance_for_search:
{topic_guidance}

present_snippets_with_ids:
{present_snippets_with_ids}
""".strip()

# Evidence retrieval field names used by prompt builders.
FIELD_CLAIM = "claim"
FIELD_TOPIC = "topic"
FIELD_DISAMBIGUATION_TEXT = "response_context"
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
NEI_LABEL = "NOT_ENOUGH_INFORMATION"


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
    - response_context: str
    - current_round_snippets: list[dict] with candidate_id/url/title/content

    Optional input fields:
    - existing_claim_snippets: list[dict] with snippet_id/url/title/content
    """
    claim = str(snippet_filter_input.get(FIELD_CLAIM) or "").strip()
    response_context = _get_disambiguation_text(snippet_filter_input)
    current_round_snippets = snippet_filter_input.get(FIELD_CURRENT_ROUND_SNIPPETS) or []
    existing_claim_snippets = snippet_filter_input.get("existing_claim_snippets") or []

    snippets_json = json.dumps(list(current_round_snippets), ensure_ascii=False, indent=2)
    existing_claim_snippets_json = json.dumps(list(existing_claim_snippets), ensure_ascii=False, indent=2)
    user_prompt = SNIPPET_FILTER_USER_PROMPT_TEMPLATE.format(
        claim=claim,
        response_context=response_context,
        current_round_snippets=snippets_json,
        existing_claim_snippets=existing_claim_snippets_json,
    )
    return SNIPPET_FILTER_SYSTEM_PROMPT_TEMPLATE, user_prompt


def build_next_search_or_answer_prompt(next_input: Mapping[str, Any]) -> tuple[str, str]:
    """Build NEXT_SEARCH_OR_ANSWER prompts for iterative retrieval.

    Required input fields:
    - claim: str
    - response_context: str
    - search_history: list[dict] with round_index/search_query/newly_added_present_snippet_ids/newly_added_present_snippets
    """
    claim = str(next_input.get(FIELD_CLAIM) or "").strip()
    response_context = _get_disambiguation_text(next_input)
    search_history = next_input.get("search_history") or []
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
        factual_label=FACTUAL_LABEL,
        response_context=response_context,
        topic_guidance=topic_guidance,
        search_history=json.dumps(list(search_history), ensure_ascii=False, indent=2),
        last_step_feedback_block=last_step_feedback_block,
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


def build_topic_grounding_evidence_extract_with_answer_prompt(payload: Mapping[str, Any]) -> tuple[str, str]:
    topic_id = str(payload.get("topic_id") or "").strip()
    query = str(payload.get("query") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    pages = payload.get("pages") or []
    system_prompt = TOPIC_GROUNDING_EVIDENCE_EXTRACT_WITH_ANSWER_SYSTEM_PROMPT_TEMPLATE
    user_prompt = TOPIC_GROUNDING_EVIDENCE_EXTRACT_WITH_ANSWER_USER_PROMPT_TEMPLATE.format(
        topic_id=topic_id,
        query=query,
        answer=answer,
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
    - response_context: str
    - present_snippets_with_ids: list[dict] with snippet_id/url/title/content
    """
    claim = str(must_input.get(FIELD_CLAIM) or "").strip()
    response_context = _get_disambiguation_text(must_input)
    topic_guidance = str(must_input.get("topic_guidance") or "").strip()
    present_snippets_with_ids = must_input.get(FIELD_PRESENT_SNIPPETS_WITH_IDS) or []

    system_prompt = MUST_HAVE_ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        factual_label=FACTUAL_LABEL,
        non_factual_label=NON_FACTUAL_LABEL,
        nei_label=NEI_LABEL,
    )
    user_prompt = MUST_HAVE_ANSWER_USER_PROMPT_TEMPLATE.format(
        claim=claim,
        response_context=response_context,
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
