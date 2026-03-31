from dataclasses import dataclass


@dataclass
class AntiObsessionConfig:
    max_surface_rate: float = 0.3  # max fraction of context slots a single topic can occupy
    min_effective_weight: float = 0.05  # prune below this weight
    one_time_query_threshold: int = 1  # topics mentioned only once are "one-time queries"
    one_time_decay_multiplier: float = 0.3  # extra decay multiplier for one-time mentions


class AntiObsessionFilter:
    """Prevents any single topic from dominating context."""

    def __init__(self, config: AntiObsessionConfig | None = None) -> None:
        self.config = config or AntiObsessionConfig()

    def apply(
        self,
        memories: list[dict],  # list of {topic, weight, frequency, ...}
        max_memories: int = 20,
    ) -> list[dict]:
        # Apply one-time query penalty
        filtered = []
        for mem in memories:
            w = mem.get("weight", 1.0)
            freq = mem.get("frequency", 1)
            if not mem.get("is_pinned") and freq <= self.config.one_time_query_threshold:
                w *= self.config.one_time_decay_multiplier
                mem = {**mem, "weight": w, "_one_time_penalized": True}
            filtered.append(mem)

        # Remove below threshold
        filtered = [m for m in filtered if m.get("weight", 0) >= self.config.min_effective_weight]

        # Sort by weight descending
        filtered.sort(key=lambda m: m.get("weight", 0), reverse=True)

        # Cap per-topic dominance: no topic group > max_surface_rate * max_memories
        max_per_topic = max(1, int(self.config.max_surface_rate * max_memories))
        topic_counts: dict[str, int] = {}
        result = []
        for mem in filtered:
            topic = mem.get("topic", "")
            count = topic_counts.get(topic, 0)
            if count < max_per_topic:
                result.append(mem)
                topic_counts[topic] = count + 1

        return result[:max_memories]
