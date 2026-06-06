from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ModelSequenceId = int  # model response index, e.g. 1, 2
ClaimSequenceId = int  # claim index within one response, e.g. 1, 2
TopicId = int  # topic group id, e.g. 1, 2
TopicName = str  # topic label, e.g. "Apple 2023 financials"


@dataclass(frozen=True)
class ClaimRef:
    """One atomic claim, keyed by (model_sequence_id, atomic_claim_sequence_number).

    Example::

        ClaimRef(
            model_sequence_id=1,
            atomic_claim_sequence_number=2,
            atomic_claim="Apple was founded in 1976.",
        )
    """

    model_sequence_id: int
    atomic_claim_sequence_number: int
    atomic_claim: str


class SplitClaimRegistry:
    """Stores split-stage claims per model for later grouping / verification lookup.

    Internal shape after writes::

        _claims_by_model = {
            1: [ClaimRef(...), ClaimRef(...)],
            2: [ClaimRef(...)],
        }
    """

    def __init__(self) -> None:
        """Create an empty registry (no models stored yet)."""
        self._claims_by_model: dict[int, list[ClaimRef]] = {}

    def add_model_claims(self, model_sequence_id: int, claims: Iterable[str]) -> list[dict]:
        """Register split claims for one model; overwrite prior data for the same model id.

        Returns JSON-ready dicts (empty list if every claim is blank)::

            [
                {"atomic_claim_sequence_number": 1, "atomic_claim": "Apple was founded in 1976."},
                {"atomic_claim_sequence_number": 2, "atomic_claim": "Apple is headquartered in Cupertino."},
            ]
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
        """Return all ClaimRef entries for one model (copy; mutating the list does not affect storage).

        Example return for model 1::

            [
                ClaimRef(1, 1, "Apple was founded in 1976."),
                ClaimRef(1, 2, "Apple is headquartered in Cupertino."),
            ]

        Returns ``[]`` if that model was never registered.
        """
        return list(self._claims_by_model.get(model_sequence_id, []))

    def get_claim_text(self, model_sequence_id: int, claim_sequence_number: int) -> str | None:
        """Look up claim text by model id + claim sequence number.

        Returns ``"Apple was founded in 1976."`` when found; ``None`` when missing.
        """
        for ref in self._claims_by_model.get(model_sequence_id, []):
            if ref.atomic_claim_sequence_number == claim_sequence_number:
                return ref.atomic_claim
        return None


class QueryTopicRegistry:
    """Per-query topic ledger: records topic names and which claim belongs to which topic.

    Does not run grouping or LLM calls — only stores results from upstream.

    After updates, public fields look like::

        topic_id_to_name = {1: "NVIDIA founding facts", 2: "GPU history"}
        claim_topic_mapping = {
            1: [(1, 1), (2, 2)],  # model 1: claim 1 -> topic 1, claim 2 -> topic 2
            2: [(1, 2), (2, 2)],
        }
    """

    def __init__(self, model_number: int) -> None:
        """Create a query-level registry; ``model_number`` is the expected model count (>= 0)."""
        self.model_number: int = max(0, int(model_number))
        self.processed_model_number: int = 0
        self.topic_id_to_name: dict[TopicId, TopicName] = {}
        self.claim_topic_mapping: dict[ModelSequenceId, list[tuple[ClaimSequenceId, TopicId]]] = {}

    def register_model_groups(self, model_sequence_id: int, groups: list[dict]) -> None:
        """Merge one model's grouping output into topic / claim mappings.
        Returns nothing. Example ``groups`` input::

            [
                {
                    "topic_group_id": 1,
                    "topic_group_name": "NVIDIA founding facts",
                    "atomic_claim_sequence_number": [1, 3],
                },
                {
                    "topic_group_id": 2,
                    "topic_group_name": "GPU history",
                    "atomic_claim_sequence_number": [2],
                },
            ]

        Updates ``topic_id_to_name`` and ``claim_topic_mapping[model_sequence_id]``; bumps
        ``processed_model_number`` the first time a new ``model_sequence_id`` is seen.
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
        """Return the topic name for a topic id.

        Returns ``"NVIDIA founding facts"`` when registered; ``None`` if topic id is unknown.
        """
        return self.topic_id_to_name.get(int(topic_id))

    def get_topic_for_claim(self, model_sequence_id: int, claim_sequence_number: int) -> int | None:
        """Return which topic id a claim belongs to.

        Returns ``2`` when model 1 claim 2 maps to topic 2; ``None`` when unmapped.
        """
        mapping = self.claim_topic_mapping.get(int(model_sequence_id), [])
        target = int(claim_sequence_number)
        for claim_seq, topic_id in mapping:
            if claim_seq == target:
                return topic_id
        return None

    def get_model_claim_topic_pairs(self, model_sequence_id: int) -> list[tuple[int, int]]:
        """Return all (claim_sequence_id, topic_id) pairs for one model (copy).

        Example return for model 1::

            [(1, 1), (2, 2)]

        Returns ``[]`` if that model has no mapping.
        """
        return list(self.claim_topic_mapping.get(int(model_sequence_id), []))

    def get_prior_topic_map(self) -> dict[str, str]:
        """Build topic summary for prompts (keys prefixed with ``T``).

        Example return::

            {"T1": "NVIDIA founding facts", "T2": "GPU history"}

        Returns ``{}`` when no topics are registered yet.
        """
        return {f"T{topic_id}": topic_name for topic_id, topic_name in sorted(self.topic_id_to_name.items())}
