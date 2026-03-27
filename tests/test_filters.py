from llm_memory_decay.filters import AntiObsessionFilter, AntiObsessionConfig


def make_memories(
    n: int, topic: str, weight: float, frequency: int = 1
) -> list[dict]:
    return [
        {
            "id": str(i),
            "content": f"mem {i}",
            "topic": topic,
            "weight": weight,
            "frequency": frequency,
        }
        for i in range(n)
    ]


def test_one_time_query_penalized():
    f = AntiObsessionFilter(AntiObsessionConfig(one_time_decay_multiplier=0.1))
    memories = [
        {
            "id": "1",
            "content": "one-off question",
            "topic": "obscure",
            "weight": 1.0,
            "frequency": 1,
        }
    ]
    result = f.apply(memories, max_memories=10)
    # Should be penalized (weight reduced) or filtered out
    if result:
        assert result[0]["weight"] < 1.0


def test_anti_obsession_caps_per_topic():
    f = AntiObsessionFilter(AntiObsessionConfig(max_surface_rate=0.3))
    # 10 memories all about the same topic
    memories = make_memories(10, "dominant_topic", weight=0.9, frequency=5)
    result = f.apply(memories, max_memories=10)
    dominant_count = sum(1 for m in result if m["topic"] == "dominant_topic")
    # Should be capped at 30% of 10 = 3
    assert dominant_count <= 3


def test_below_threshold_pruned():
    f = AntiObsessionFilter(AntiObsessionConfig(min_effective_weight=0.1))
    memories = [
        {"id": "1", "content": "strong", "topic": "a", "weight": 0.9, "frequency": 5},
        {"id": "2", "content": "weak", "topic": "b", "weight": 0.01, "frequency": 5},
    ]
    result = f.apply(memories, max_memories=10)
    contents = [m["content"] for m in result]
    assert "strong" in contents
    assert "weak" not in contents


def test_empty_memories():
    f = AntiObsessionFilter()
    result = f.apply([], max_memories=10)
    assert result == []


def test_sorted_by_weight_descending():
    f = AntiObsessionFilter()
    memories = [
        {"id": "1", "content": "low", "topic": "a", "weight": 0.2, "frequency": 5},
        {"id": "2", "content": "high", "topic": "b", "weight": 0.9, "frequency": 5},
        {"id": "3", "content": "mid", "topic": "c", "weight": 0.5, "frequency": 5},
    ]
    result = f.apply(memories, max_memories=10)
    weights = [m["weight"] for m in result]
    assert weights == sorted(weights, reverse=True)


def test_max_memories_respected():
    f = AntiObsessionFilter()
    memories = make_memories(30, "topic", weight=0.8, frequency=5)
    result = f.apply(memories, max_memories=5)
    assert len(result) <= 5


def test_one_time_penalized_flag_set():
    f = AntiObsessionFilter(
        AntiObsessionConfig(
            one_time_decay_multiplier=0.5,
            min_effective_weight=0.0,
        )
    )
    memories = [
        {"id": "1", "content": "one-off", "topic": "a", "weight": 1.0, "frequency": 1}
    ]
    result = f.apply(memories, max_memories=10)
    assert result[0].get("_one_time_penalized") is True
