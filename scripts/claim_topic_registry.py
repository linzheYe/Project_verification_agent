from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# 类型别名（为了让注释和字段含义更直观）
# - ModelSequenceId: 第几个模型回复（response_id），例如 1, 2, 3
# - ClaimSequenceId: 该模型回复内第几条 claim（claim_in_response_id），例如 1, 2, 3
# - TopicId: 主题组编号（int），例如 1, 2
# - TopicName: 主题名，例如 "Apple 2023 financials"
ModelSequenceId = int
ClaimSequenceId = int
TopicId = int
TopicName = str


@dataclass(frozen=True)
class ClaimRef:
    """轻量 claim 引用。

    一个 claim 在 query 内由 `(model_sequence_id, atomic_claim_sequence_number)` 唯一定位。
    """

    model_sequence_id: int
    atomic_claim_sequence_number: int
    atomic_claim: str


class SplitClaimRegistry:
    """保存拆分阶段结果，并对外暴露 claim 查询接口。

    作用：
    - 记住每个模型回复被拆成了哪些 claim。
    - 提供稳定接口供后续 grouping / verification 阶段读取。
    """

    def __init__(self) -> None:
        """初始化空 registry。"""
        self._claims_by_model: dict[int, list[ClaimRef]] = {}

    def add_model_claims(self, model_sequence_id: int, claims: Iterable[str]) -> list[dict]:
        """写入一个模型的拆分结果，并返回 JSON 可写格式。

        参数：
        - model_sequence_id: 模型回复序号（response_id）。
        - claims: 拆分得到的 claim 文本序列。
        """
        refs: list[ClaimRef] = []
        for i, claim in enumerate(claims, start=1):
            text = str(claim).strip()
            if not text:
                continue
            refs.append(
                ClaimRef(
                    model_sequence_id=model_sequence_id,
                    atomic_claim_sequence_number=i,
                    atomic_claim=text,
                )
            )

        self._claims_by_model[model_sequence_id] = refs
        return [
            {
                "atomic_claim_sequence_number": ref.atomic_claim_sequence_number,
                "atomic_claim": ref.atomic_claim,
            }
            for ref in refs
        ]

    def get_claims_for_model(self, model_sequence_id: int) -> list[ClaimRef]:
        """返回某个模型下的 claim 列表。"""
        return list(self._claims_by_model.get(model_sequence_id, []))

    def get_claim_text(self, model_sequence_id: int, claim_sequence_number: int) -> str | None:
        """按双序号查询 claim 文本。"""
        for ref in self._claims_by_model.get(model_sequence_id, []):
            if ref.atomic_claim_sequence_number == claim_sequence_number:
                return ref.atomic_claim
        return None


class QueryTopicRegistry:
    """每个 query 一个 topic registry，支持随模型推进增量更新。

    - model_number
    - processed_model_number
    - topic_id_to_name: dict[topic_id(int), topic_name(str)]
    - claim_topic_mapping: dict[model_sequence_id, list[(claim_sequence_id, topic_id)]]
    说明：该类只负责记录，不负责 topic 生成/分组决策。
    """

    def __init__(self, model_number: int) -> None:
        """创建 query 级 registry。"""
        self.model_number: int = max(0, int(model_number))
        self.processed_model_number: int = 0
        # topic_id_to_name:
        # - key(int): topic_id，如 1, 2
        # - value(str): topic_name，如 "NVIDIA founding facts"
        # 示例: {1: "NVIDIA founding facts", 2: "GPU history"}
        self.topic_id_to_name: dict[TopicId, TopicName] = {}

        # claim_topic_mapping:
        # - key(int): model_sequence_id
        # - value(list[tuple[int,int]]): [(claim_sequence_id, topic_id), ...]
        # 也就是：每个模型回复里，每条 claim 属于哪个 topic（按 pair 记录）。
        # 示例:
        # {
        #   1: [(1, 1), (2, 2)],   # 模型1: claim1->topic1, claim2->topic2
        #   2: [(1, 2), (2, 2)]    # 模型2: claim1->topic2, claim2->topic2
        # }
        self.claim_topic_mapping: dict[ModelSequenceId, list[tuple[ClaimSequenceId, TopicId]]] = {}

    def register_model_groups(self, model_sequence_id: int, groups: list[dict]) -> None:
        """写入一个模型的分组结果，并维护 query 级状态。

        约定：`groups` 已由外部模块完成格式校验与 topic_id 解析。
        本类不做 topic 生成，也不做 LLM 输出纠错。
        """
        if int(model_sequence_id) not in self.claim_topic_mapping:
            self.processed_model_number += 1

        model_map: dict[int, int] = {
            claim_seq: topic_id
            for claim_seq, topic_id in self.claim_topic_mapping.get(int(model_sequence_id), [])
        }

        for group in groups:
            try:
                topic_id = int(group.get("topic_group_id"))
            except Exception:
                continue
            topic_name = str(group.get("topic_group_name") or "").strip()
            seq_nums = group.get("atomic_claim_sequence_number") or []

            # 保持 topic_id -> topic_name 稳定：默认沿用第一次出现的名字。
            if topic_id not in self.topic_id_to_name:
                self.topic_id_to_name[topic_id] = topic_name
            elif not self.topic_id_to_name[topic_id] and topic_name:
                self.topic_id_to_name[topic_id] = topic_name

            if not isinstance(seq_nums, list):
                continue
            for seq in seq_nums:
                try:
                    seq_int = int(seq)
                except Exception:
                    continue
                model_map[seq_int] = topic_id
        self.claim_topic_mapping[int(model_sequence_id)] = sorted(model_map.items(), key=lambda x: x[0])

    def get_topic_name(self, topic_id: int) -> str | None:
        """按 topic_id 查询 topic_name。"""
        return self.topic_id_to_name.get(int(topic_id))

    def get_topic_for_claim(self, model_sequence_id: int, claim_sequence_number: int) -> int | None:
        """查询某条 claim 归属的 topic_id。"""
        mapping = self.claim_topic_mapping.get(int(model_sequence_id), [])
        target = int(claim_sequence_number)
        for claim_seq, topic_id in mapping:
            if claim_seq == target:
                return topic_id
        return None

    def get_model_claim_topic_pairs(self, model_sequence_id: int) -> list[tuple[int, int]]:
        """返回某个模型的 claim-topic 对列表：[(claim_sequence_id, topic_id), ...]。"""
        return list(self.claim_topic_mapping.get(int(model_sequence_id), []))

    def get_prior_topic_map(self) -> dict[str, str]:
        """返回供 prompt 使用的 topic_id -> topic_name（key 格式为 'Tn'）。"""
        return {f"T{topic_id}": topic_name for topic_id, topic_name in sorted(self.topic_id_to_name.items())}
