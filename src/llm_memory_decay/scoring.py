from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TopicScore:
    topic: str
    frequency: int = 0
    last_seen: datetime | None = None
    importance: float = 1.0  # user-set or inferred
    is_pinned: bool = False  # pinned memories never decay


class ImportanceScorer:
    """Scores topics by recurrence and user signals."""

    def __init__(self) -> None:
        self._topic_scores: dict[str, TopicScore] = {}

    def record_mention(
        self,
        topic: str,
        at: datetime | None = None,
        importance: float | None = None,
    ) -> None:
        if at is None:
            at = datetime.utcnow()
        if topic not in self._topic_scores:
            self._topic_scores[topic] = TopicScore(topic=topic)
        score = self._topic_scores[topic]
        score.frequency += 1
        score.last_seen = at
        if importance is not None:
            score.importance = importance

    def get_frequency_weight(self, topic: str, max_frequency: int = 10) -> float:
        """Normalize frequency to [0.1, 1.0]."""
        if topic not in self._topic_scores:
            return 0.1
        freq = self._topic_scores[topic].frequency
        return min(1.0, 0.1 + 0.9 * (freq / max_frequency))

    def get_importance(self, topic: str) -> float:
        if topic not in self._topic_scores:
            return 1.0
        return self._topic_scores[topic].importance

    def is_pinned(self, topic: str) -> bool:
        return self._topic_scores.get(topic, TopicScore(topic=topic)).is_pinned

    def pin_topic(self, topic: str) -> None:
        if topic not in self._topic_scores:
            self._topic_scores[topic] = TopicScore(topic=topic)
        self._topic_scores[topic].is_pinned = True

    def all_topics(self) -> list[TopicScore]:
        return list(self._topic_scores.values())
