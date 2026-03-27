from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid

from .decay import DecayStrategy, compute_decay_weight
from .scoring import ImportanceScorer
from .filters import AntiObsessionFilter, AntiObsessionConfig


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    topic: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    frequency: int = 1
    importance: float = 1.0
    is_pinned: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "topic": self.topic,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "frequency": self.frequency,
            "importance": self.importance,
            "is_pinned": self.is_pinned,
        }


class MemoryStore:
    """
    Core memory store with principled decay.

    Replaces naive "remember everything forever" approaches with:
    - Recency weighting (exponential/linear/step decay)
    - Topic importance scoring (frequency + user signals)
    - Anti-obsession filtering (prevents one-off topics from dominating)
    """

    def __init__(
        self,
        decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL,
        half_life_days: float = 30.0,
        max_age_days: float = 90.0,
        anti_obsession: AntiObsessionConfig | None = None,
        max_memories: int = 50,
    ) -> None:
        self.decay_strategy = decay_strategy
        self.half_life_days = half_life_days
        self.max_age_days = max_age_days
        self.max_memories = max_memories
        self._entries: list[MemoryEntry] = []
        self._scorer = ImportanceScorer()
        self._filter = AntiObsessionFilter(anti_obsession or AntiObsessionConfig())

    def add(
        self,
        content: str,
        topic: str = "",
        importance: float = 1.0,
        created_at: datetime | None = None,
        metadata: dict | None = None,
        pinned: bool = False,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            topic=topic,
            created_at=created_at or datetime.utcnow(),
            importance=importance,
            metadata=metadata or {},
            is_pinned=pinned,
        )
        self._entries.append(entry)
        self._scorer.record_mention(topic, at=entry.created_at, importance=importance)
        if pinned:
            self._scorer.pin_topic(topic)
        return entry

    def get_weighted_memories(self, now: datetime | None = None) -> list[dict]:
        if now is None:
            now = datetime.utcnow()

        result = []
        for entry in self._entries:
            if entry.is_pinned:
                weight = 1.0
            else:
                decay_w = compute_decay_weight(
                    entry.created_at,
                    strategy=self.decay_strategy,
                    half_life_days=self.half_life_days,
                    max_age_days=self.max_age_days,
                    now=now,
                )
                importance_w = entry.importance
                freq_w = self._scorer.get_frequency_weight(entry.topic)
                weight = decay_w * importance_w * freq_w

            result.append(
                {
                    "id": entry.id,
                    "content": entry.content,
                    "topic": entry.topic,
                    "weight": weight,
                    "frequency": entry.frequency,
                    "is_pinned": entry.is_pinned,
                    "created_at": entry.created_at,
                    "metadata": entry.metadata,
                }
            )

        return self._filter.apply(result, max_memories=self.max_memories)

    def get_context_memories(
        self, top_k: int = 10, now: datetime | None = None
    ) -> list[dict]:
        """Return top-k memories by effective weight, suitable for context injection."""
        weighted = self.get_weighted_memories(now=now)
        return sorted(weighted, key=lambda m: m["weight"], reverse=True)[:top_k]

    def forget(self, memory_id: str) -> bool:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.id != memory_id]
        return len(self._entries) < before

    def pin(self, memory_id: str) -> bool:
        for entry in self._entries:
            if entry.id == memory_id:
                entry.is_pinned = True
                self._scorer.pin_topic(entry.topic)
                return True
        return False

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"MemoryStore(entries={len(self._entries)}, strategy={self.decay_strategy})"
