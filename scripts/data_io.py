from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

VALID_FINAL_ANSWER_LABELS = {
    "SUPPORTED",
    "REFUTED",
    "NOT_ENOUGH_INFORMATION",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，返回对象列表。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """将对象列表写入 JSONL 文件（逐行一个 JSON 对象）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_source_records(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 source 行式数据组织成 pipeline 统一输入结构。

    输入（每行）示例：
    - id
    - query
    - model
    - model_answer

    输出（每个 sample 一条）：
    {
      "sample": {"id":..., "query":...},
      "model_outputs": {
        "1": {"model_sequence_id":1, "model_name":"...", "model_answer":"..."},
        "2": {...}
      }
    }
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    seq_counter: dict[tuple[str, str], int] = {}

    for row in source_rows:
        sid = str(row.get("id") or "").strip()
        query = str(row.get("query") or "").strip()
        if not sid or not query:
            continue

        key = (sid, query)
        if key not in grouped:
            grouped[key] = {"sample": {"id": sid, "query": query}, "model_outputs": {}}
            seq_counter[key] = 0

        seq_counter[key] += 1
        seq = seq_counter[key]
        grouped[key]["model_outputs"][str(seq)] = {
            "model_sequence_id": seq,
            "model_name": str(row.get("model") or "").strip(),
            "model_answer": str(row.get("model_answer") or "").strip(),
        }

    return list(grouped.values())


def parse_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出中解析 JSON object。

    兼容两类输入：
    - 纯 JSON 文本
    - 带 ```json code fence 的文本
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        obj = json.loads(text)
    except Exception:
        l = text.find("{")
        r = text.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise ValueError("no JSON object found")
        obj = json.loads(text[l : r + 1])

    if not isinstance(obj, dict):
        raise ValueError("JSON root must be object")
    return obj


def _extract_json_object_substring(text: str) -> str:
    l = text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError("no JSON object found")
    return text[l : r + 1]


def _parse_json_object_strict(text: str) -> tuple[dict[str, Any], str]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")

    normalized = text
    if normalized.startswith("```"):
        normalized = normalized.strip("`").strip()
        if normalized.startswith("json"):
            normalized = normalized[4:].strip()

    try:
        obj = json.loads(normalized)
        if not isinstance(obj, dict):
            raise ValueError("JSON root must be object")
        return obj, "strict_json"
    except Exception:
        candidate = _extract_json_object_substring(normalized)
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("JSON root must be object")
        return obj, "strict_brace_extract"


def _try_json_repair_parse(text: str) -> tuple[dict[str, Any], str]:
    try:
        import json_repair  # type: ignore[import-not-found]
    except Exception as e:
        raise ValueError(f"json_repair_unavailable: {e}") from e

    obj = json_repair.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("json_repair_root_not_object")
    return obj, "json_repair"


def _find_key_value_anchor(text: str, key: str) -> int:
    pattern = re.compile(rf'(?i)["\']?{re.escape(key)}["\']?\s*[:=]')
    m = pattern.search(text)
    return -1 if m is None else m.end()


def _parse_value_token(text: str, start_idx: int) -> str:
    n = len(text)
    i = max(0, start_idx)
    while i < n and text[i].isspace():
        i += 1
    if i >= n:
        return ""

    quote = text[i]
    if quote in {"'", '"'}:
        i += 1
        buf: list[str] = []
        escape = False
        while i < n:
            ch = text[i]
            if escape:
                buf.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                break
            else:
                buf.append(ch)
            i += 1
        return "".join(buf).strip()

    j = i
    while j < n and text[j] not in {",", "\n", "\r", "}"}:
        j += 1
    return text[i:j].strip()


def _extract_answer_like_fields(text: str) -> dict[str, Any]:
    raw = str(text or "")
    if not raw.strip():
        raise ValueError("empty text")

    final_pos = _find_key_value_anchor(raw, "final_answer")
    if final_pos < 0:
        raise ValueError("final_answer_not_found_in_text")
    final_answer = _parse_value_token(raw, final_pos).strip().upper()
    if final_answer not in VALID_FINAL_ANSWER_LABELS:
        raise ValueError("final_answer_not_in_valid_label_set")

    reason = ""
    reason_pos = _find_key_value_anchor(raw, "reason")
    if reason_pos >= 0:
        reason = _parse_value_token(raw, reason_pos).strip()
    if not reason:
        reasoning_pos = _find_key_value_anchor(raw, "reasoning")
        if reasoning_pos >= 0:
            reason = _parse_value_token(raw, reasoning_pos).strip()

    evidence_ids = re.findall(r"\bS\d+\b", raw, flags=re.IGNORECASE)
    evidence_ids = [sid.upper() for sid in evidence_ids]
    unique_ids: list[str] = []
    seen: set[str] = set()
    for sid in evidence_ids:
        if sid in seen:
            continue
        seen.add(sid)
        unique_ids.append(sid)

    return {
        "final_answer": final_answer,
        "reason": reason,
        "evidence_snippet_ids": unique_ids,
    }


def parse_json_object_robust(text: str) -> tuple[dict[str, Any], str]:
    """Parse LLM output into JSON object with layered fallbacks.

    Returns:
    - object: Parsed object.
    - parse_mode: One of strict_json / strict_brace_extract / json_repair / key_extract.
    """
    strict_error = None
    try:
        return _parse_json_object_strict(text)
    except Exception as e:
        strict_error = e

    repair_error = None
    try:
        return _try_json_repair_parse(text)
    except Exception as e:
        repair_error = e

    try:
        extracted = _extract_answer_like_fields(text)
        return extracted, "key_extract"
    except Exception as e:
        raise ValueError(
            f"robust_parse_failed: strict={strict_error}; repair={repair_error}; extract={e}"
        ) from e


def parse_atomic_claim_lines(text: str) -> list[str]:
    """将 split 阶段的逐行文本结果解析为 claim 列表。

    解析规则：
    - 忽略空行。
    - 清理常见前缀（数字编号、短横线等）。
    - 去重但保持原顺序。
    """
    text = (text or "").strip()
    if not text or text.lower() == "no verifiable claim":
        return []

    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. )")
        if line:
            claims.append(line)

    unique: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if claim not in seen:
            unique.append(claim)
            seen.add(claim)
    return unique
