#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_io import parse_json_object, read_jsonl, write_jsonl
from scripts.llm_api import LLMClient
from scripts.prompt_template import build_rubrics_verdict_prompt

INPUT_RUBRICS_JSONL = Path("data/rubrics_input.jsonl")  # rubrics源数据(相对项目根目录)
EVIDENCE_RESULTS_PATH = Path("data/runs/example/evidence_results.jsonl")  # 已有retrieve输出
OUTPUT_DIR = Path("data/runs/example")  # 输出目录
OUTPUT_FILE = "rubrics_verdict_result.jsonl"  # 输出文件名
REPORT_FILE = "rubrics_verdict_report.json"  # 运行报告文件名

START_LINE = 1  # 从rubrics输入第几行开始(1-based)
NUM_LINES = 50  # 使用多少行rubrics输入
RUBRICS_VERDICT_MODEL = "openai/gpt-5.4-nano"  # rubrics判分模型


def _slice_rows(rows: list[dict[str, Any]], start_line: int, num_lines: int) -> list[dict[str, Any]]:
    s = max(0, start_line - 1)
    e = s + max(0, num_lines)
    return rows[s:e]


def _sample_id_sort_key(sample_id: str) -> tuple[int, str]:
    s = str(sample_id or "").strip()
    if s.startswith("simpleqa_"):
        tail = s.split("_", 1)[1]
        if tail.isdigit():
            return (int(tail), s)
    return (10**9, s)


def _normalize_verdict_rows(evidence_rows: list[dict[str, Any]], sample_id: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in evidence_rows:
        if str(row.get("sample_id") or "").strip() != sample_id:
            continue
        out.append(
            {
                "claim": str(row.get("atomic_claim") or "").strip(),
                "verdict": str(row.get("auto_verdict") or "").strip(),
                "reason": str(row.get("verdict_rationale") or "").strip(),
            }
        )
    return out


def main() -> None:
    input_rubrics_path = PROJECT_ROOT / INPUT_RUBRICS_JSONL
    evidence_results_path = PROJECT_ROOT / EVIDENCE_RESULTS_PATH
    output_dir = PROJECT_ROOT / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILE
    report_path = output_dir / REPORT_FILE

    if not input_rubrics_path.exists():
        raise FileNotFoundError(f"rubrics input not found: {input_rubrics_path}")
    if not evidence_results_path.exists():
        raise FileNotFoundError(f"evidence_results not found: {evidence_results_path}")

    rubrics_rows_all = read_jsonl(input_rubrics_path)
    rubrics_rows = _slice_rows(rubrics_rows_all, START_LINE, NUM_LINES)
    if not rubrics_rows:
        raise RuntimeError("selected rubrics rows are empty; check START_LINE/NUM_LINES")

    source_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rubrics_rows, start=1):
        global_idx = START_LINE + i - 1
        source_rows.append(
            {
                "id": f"simpleqa_{global_idx}",
                "query": str(row.get("original_question") or row.get("query") or "").strip(),
                "model_answer": str(row.get("answer") or row.get("model_answer") or "").strip(),
                "rubrics_correct": row.get("rubrics_correct"),
                "rubrics_wrong": row.get("rubrics_wrong"),
            }
        )

    source_by_id = {str(x.get("id") or "").strip(): x for x in source_rows}
    evidence_rows = read_jsonl(evidence_results_path)
    evidence_sample_ids = sorted(
        {
            str(r.get("sample_id") or "").strip()
            for r in evidence_rows
            if str(r.get("sample_id") or "").strip()
        },
        key=_sample_id_sort_key,
    )

    llm = LLMClient(provider="openrouter", model=RUBRICS_VERDICT_MODEL, timeout_seconds=45, max_retries=1)
    output_rows: list[dict[str, Any]] = []
    missing_sample_ids: list[str] = []

    for sample_id in evidence_sample_ids:
        src = source_by_id.get(sample_id)
        if src is None:
            output_rows.append({"sample_id": sample_id, "status": "error", "error": "sample_id_not_found_in_rubrics_input"})
            missing_sample_ids.append(sample_id)
            continue

        verdict_rows = _normalize_verdict_rows(evidence_rows, sample_id)
        sys_prompt, user_prompt = build_rubrics_verdict_prompt(
            {
                "verdict_of_each_claim": verdict_rows,
                "original_response": src.get("model_answer"),
                "rubrics_correct": src.get("rubrics_correct"),
                "rubrics_wrong": src.get("rubrics_wrong"),
            }
        )

        try:
            raw = llm.call(sys_prompt, user_prompt, temperature=0.2)
            obj = parse_json_object(raw)
            output_rows.append({"sample_id": sample_id, "status": "ok", "result": obj})
        except Exception as e:
            output_rows.append({"sample_id": sample_id, "status": "error", "error": f"rubrics_scoring_error: {e}"})

    write_jsonl(output_path, output_rows)
    ok_count = sum(1 for r in output_rows if str(r.get("status")) == "ok")
    error_count = len(output_rows) - ok_count
    report = {
        "status": "ok",
        "input_rubrics_jsonl": str(input_rubrics_path),
        "evidence_results_path": str(evidence_results_path),
        "start_line": START_LINE,
        "num_lines": NUM_LINES,
        "rubrics_model": RUBRICS_VERDICT_MODEL,
        "output_file": str(output_path),
        "total_scored": len(output_rows),
        "ok_count": ok_count,
        "error_count": error_count,
        "missing_sample_id_count": len(missing_sample_ids),
        "missing_sample_ids": missing_sample_ids,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_path": str(output_path), "report_path": str(report_path), "count": len(output_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
