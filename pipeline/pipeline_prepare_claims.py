#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from scripts.data_io import (
    build_source_records,
    parse_atomic_claim_lines,
    parse_json_object,
    read_jsonl,
    write_jsonl,
)
from scripts.llm_api import LLMClient, LLMConnectivityError
from scripts.prompt_template import build_clean_prompt, build_group_prompt, build_split_prompt
from scripts.claim_topic_registry import SplitClaimRegistry


class FatalPrepareClaimsError(RuntimeError):
    """Reserved for truly unrecoverable prepare-stage failures."""


class PrepareClaimsPipeline:
    """最小可复用 pipeline：clean -> split -> group。

    对外暴露两个运行期 registry：
    - split_registries: 记录每个 query 下“模型回复 -> claim 序列”的映射。
    - group 输出改为 question 单组，不再使用 topic 相关状态。
    """

    def __init__(
        self,
        source_path: Path,
        clean_path: Path,
        split_path: Path,
        group_path: Path,
        llm_client: LLMClient | None = None,
    ) -> None:
        """初始化路径配置与 LLM 客户端。"""
        self.source_path = source_path
        self.clean_path = clean_path
        self.split_path = split_path
        self.group_path = group_path
        self.llm = llm_client or LLMClient()

        # 键为 (sample_id, query)
        self.split_registries: dict[tuple[str, str], SplitClaimRegistry] = {}
    @staticmethod
    def _sample_key(rec: dict[str, Any]) -> tuple[str, str]:
        """提取 sample 主键，用作 registry 的索引键。"""
        sample = rec.get("sample") or {}
        return str(sample.get("id") or ""), str(sample.get("query") or "")

    @staticmethod
    def _iter_model_outputs(rec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """按 model_sequence_id 升序迭代模型输出。"""
        return sorted(
            (rec.get("model_outputs") or {}).items(),
            key=lambda kv: int((kv[1] or {}).get("model_sequence_id") or 0),
        )

    @staticmethod
    def _ts() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def run_clean_stage(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """执行 clean：model_answer -> cleaned_model_answer。"""
        out_rows: list[dict[str, Any]] = []

        for rec in records:
            sample = rec.get("sample") or {}
            sample_id = str(sample.get("id") or "").strip()
            query = str(sample.get("query") or "").strip()
            out_model_outputs: dict[str, dict[str, Any]] = {}
            for seq_key, model_output in self._iter_model_outputs(rec):
                model_answer = str(model_output.get("model_answer") or "").strip()
                model_sequence_id = int(model_output.get("model_sequence_id") or 0)
                model_name = str(model_output.get("model_name") or "")
                item: dict[str, Any] = {
                    "model_sequence_id": model_sequence_id,
                    "model_name": model_name,
                    "raw_model_answer": model_answer,
                    "cleaned_model_answer": "",
                }

                if not model_answer:
                    item["error"] = "empty_model_answer"
                    print(
                        f"[{self._ts()}] [prepare:clean:skip] sample_id={sample_id} seq={model_sequence_id} "
                        f"model={model_name} reason=empty_model_answer",
                        flush=True,
                    )
                else:
                    try:
                        print(
                            f"[{self._ts()}] [prepare:clean:llm:start] sample_id={sample_id} seq={model_sequence_id} "
                            f"model={model_name} query={query[:120]!r} answer_chars={len(model_answer)}",
                            flush=True,
                        )
                        t0 = time.perf_counter()
                        system_prompt, user_prompt = build_clean_prompt(model_answer)
                        raw = self.llm.call(system_prompt, user_prompt, temperature=0.2)
                        parsed = parse_json_object(raw)
                        if "cleaned_response" not in parsed:
                            raise ValueError("cleaned_response missing")
                        cleaned = str(parsed.get("cleaned_response") or "").strip()
                        item["cleaned_model_answer"] = cleaned
                        print(
                            f"[{self._ts()}] [prepare:clean:llm:done] sample_id={sample_id} seq={model_sequence_id} "
                            f"model={model_name} elapsed={time.perf_counter() - t0:.2f}s cleaned_chars={len(cleaned)}",
                            flush=True,
                        )
                    except Exception as e:
                        item["error"] = f"clean_error: {e}"
                        print(
                            f"[{self._ts()}] [prepare:clean:llm:error] sample_id={sample_id} seq={model_sequence_id} "
                            f"model={model_name} error={e}",
                            flush=True,
                        )

                out_model_outputs[seq_key] = item

            out_rows.append({"sample": rec.get("sample"), "model_outputs": out_model_outputs})

        return out_rows

    def run_split_stage(self, clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """执行 split：cleaned_model_answer -> atomic_claims。

        同时将拆分结果写入 `split_registries`，供后续模块通过类接口读取。
        """
        out_rows: list[dict[str, Any]] = []

        for rec in clean_rows:
            sample_key = self._sample_key(rec)
            sample = rec.get("sample") or {}
            sample_id = str(sample.get("id") or "").strip()
            query = str(sample.get("query") or "").strip()
            split_registry = SplitClaimRegistry()

            out_model_outputs: dict[str, dict[str, Any]] = {}
            for seq_key, model_output in self._iter_model_outputs(rec):
                model_sequence_id = int(model_output.get("model_sequence_id") or 0)
                raw_model_answer = str(model_output.get("raw_model_answer") or "").strip()
                cleaned_text = str(model_output.get("cleaned_model_answer") or "").strip()
                model_name = str(model_output.get("model_name") or "")
                item: dict[str, Any] = {
                    "model_sequence_id": model_sequence_id,
                    "model_name": model_name,
                    "raw_model_answer": raw_model_answer,
                    "cleaned_model_answer": cleaned_text,
                    "atomic_claims": [],
                }

                if not cleaned_text:
                    upstream_error = str(model_output.get("error") or "").strip()
                    if upstream_error:
                        item["error"] = upstream_error
                    print(
                        f"[{self._ts()}] [prepare:split:skip] sample_id={sample_id} seq={model_sequence_id} "
                        f"model={model_name} reason={upstream_error or 'empty_cleaned_model_answer'}",
                        flush=True,
                    )
                else:
                    try:
                        print(
                            f"[{self._ts()}] [prepare:split:llm:start] sample_id={sample_id} seq={model_sequence_id} "
                            f"model={model_name} query={query[:120]!r} cleaned_chars={len(cleaned_text)}",
                            flush=True,
                        )
                        t0 = time.perf_counter()
                        system_prompt, user_prompt = build_split_prompt(cleaned_text)
                        raw = self.llm.call(system_prompt, user_prompt, temperature=0.5)
                        claims = parse_atomic_claim_lines(raw)
                        item["atomic_claims"] = split_registry.add_model_claims(model_sequence_id, claims)
                        print(
                            f"[{self._ts()}] [prepare:split:llm:done] sample_id={sample_id} seq={model_sequence_id} "
                            f"model={model_name} elapsed={time.perf_counter() - t0:.2f}s claims={len(item['atomic_claims'])}",
                            flush=True,
                        )
                    except Exception as e:
                        item["error"] = f"split_error: {e}"
                        print(
                            f"[{self._ts()}] [prepare:split:llm:error] sample_id={sample_id} seq={model_sequence_id} "
                            f"model={model_name} error={e}",
                            flush=True,
                        )

                out_model_outputs[seq_key] = item

            self.split_registries[sample_key] = split_registry
            out_rows.append({"sample": rec.get("sample"), "model_outputs": out_model_outputs})

        return out_rows

    def run_group_stage(self, split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """执行 group：atomic_claims -> question_group（每个 question 仅一组）。"""
        out_rows: list[dict[str, Any]] = []

        for rec in split_rows:
            sample_key = self._sample_key(rec)
            model_outputs = self._iter_model_outputs(rec)

            split_registry = self.split_registries.get(sample_key, SplitClaimRegistry())
            if sample_key not in self.split_registries:
                # 兜底：如果 run_group_stage 被单独调用，从 split_rows 重建 split registry。
                for _, model_output in model_outputs:
                    mid = int(model_output.get("model_sequence_id") or 0)
                    atomic_claims = model_output.get("atomic_claims") or []
                    claim_texts = [str(x.get("atomic_claim") or "").strip() for x in atomic_claims]
                    split_registry.add_model_claims(mid, claim_texts)

            out_model_outputs: dict[str, dict[str, Any]] = {}
            for seq_key, model_output in model_outputs:
                model_sequence_id = int(model_output.get("model_sequence_id") or 0)
                item: dict[str, Any] = {
                    "model_sequence_id": model_sequence_id,
                    "model_name": str(model_output.get("model_name") or ""),
                    "question_groups": [],
                }

                claim_refs = split_registry.get_claims_for_model(model_sequence_id)
                claim_texts = [c.atomic_claim for c in claim_refs]
                if not claim_texts:
                    upstream_error = str(model_output.get("error") or "").strip()
                    if upstream_error:
                        item["error"] = upstream_error
                    out_model_outputs[seq_key] = item
                    continue

                # 处理重复 claim 文本：一个文本可能对应多个序号，所以用 queue 映射。
                text_to_seq_queue: dict[str, list[int]] = {}
                for ref in claim_refs:
                    text_to_seq_queue.setdefault(ref.atomic_claim, []).append(ref.atomic_claim_sequence_number)

                try:
                    system_prompt, user_prompt = build_group_prompt(claim_texts, {})
                    raw = self.llm.call(system_prompt, user_prompt, temperature=0.2)
                    parsed = parse_json_object(raw)
                    group_name = str(parsed.get("group_name") or "").strip() or "question_group"
                    group_claims_raw = parsed.get("group_claims")
                    if not isinstance(group_claims_raw, list):
                        group_claims_raw = []
                    group_claims = [str(c).strip() for c in group_claims_raw if str(c).strip()]

                    seq_numbers: list[int] = []
                    mapped_claims: list[str] = []
                    for claim_text in group_claims:
                        queue = text_to_seq_queue.get(claim_text) or []
                        if not queue:
                            continue
                        seq_num = queue.pop(0)
                        seq_numbers.append(seq_num)
                        mapped_claims.append(claim_text)

                    if not mapped_claims:
                        # fallback: keep all split claims in original order
                        mapped_claims = [c.atomic_claim for c in claim_refs]
                        seq_numbers = [c.atomic_claim_sequence_number for c in claim_refs]

                    item["question_groups"] = [
                        {
                            "question_group_name": group_name,
                            "question_claims": mapped_claims,
                            "atomic_claim_sequence_number": seq_numbers,
                        }
                    ]
                except Exception as e:
                    self._raise_if_fatal_connectivity_error(e, stage="group")
                    item["error"] = f"group_error: {e}"

                out_model_outputs[seq_key] = item

            out_rows.append({"sample": rec.get("sample"), "model_outputs": out_model_outputs})

        return out_rows

    def get_split_registry(self, sample_id: str, query: str) -> SplitClaimRegistry | None:
        """对外接口：按 sample 查询 split registry。"""
        return self.split_registries.get((str(sample_id), str(query)))

    def run(self) -> None:
        """执行完整流水线，并写出三份 JSONL 结果。"""
        source_rows = read_jsonl(self.source_path)
        source_records = build_source_records(source_rows)

        clean_rows = self.run_clean_stage(source_records)
        write_jsonl(self.clean_path, clean_rows)

        split_rows = self.run_split_stage(clean_rows)
        write_jsonl(self.split_path, split_rows)

        group_rows = self.run_group_stage(split_rows)
        write_jsonl(self.group_path, group_rows)

        print(f"done: source={self.source_path}")
        print(f"- clean -> {self.clean_path}")
        print(f"- split -> {self.split_path}")
        print(f"- group -> {self.group_path}")


def main() -> None:
    """脚本入口：配置路径并运行 pipeline。"""
    base_dir = Path(__file__).resolve().parents[1]
    run_dir = base_dir / "data_prepare_outputs" / "runs" / "round2"

    pipeline = PrepareClaimsPipeline(
        source_path=run_dir / "source_data.jsonl",
        clean_path=run_dir / "cleaned_answer.second_try.jsonl",
        split_path=run_dir / "split_claims.second_try.jsonl",
        group_path=run_dir / "group_claims.second_try.jsonl",
        llm_client=LLMClient(provider="openrouter"),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
