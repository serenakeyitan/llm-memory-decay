from datetime import datetime, timedelta

from llm_memory_decay.decay import DecayStrategy
from llm_memory_decay.memory import MemoryStore


def make_store(**kwargs) -> MemoryStore:
    return MemoryStore(**kwargs)


def test_add_and_retrieve():
    store = make_store()
    entry = store.add("I like Python", topic="programming")
    assert len(store) == 1
    assert entry.content == "I like Python"
    assert entry.topic == "programming"


def test_weighted_memories_recent_higher():
    from llm_memory_decay.filters import AntiObsessionConfig

    now = datetime.utcnow()
    store = make_store(
        decay_strategy=DecayStrategy.EXPONENTIAL,
        half_life_days=30,
        anti_obsession=AntiObsessionConfig(min_effective_weight=0.0),
    )
    store.add("Recent memory", topic="test", created_at=now - timedelta(days=1))
    store.add("Old memory", topic="test2", created_at=now - timedelta(days=60))

    weighted = store.get_weighted_memories(now=now)
    weights_by_content = {m["content"]: m["weight"] for m in weighted}

    # Recent should have higher weight
    assert weights_by_content["Recent memory"] > weights_by_content["Old memory"]


def test_pinned_memory_never_decays():
    now = datetime.utcnow()
    store = make_store(decay_strategy=DecayStrategy.EXPONENTIAL, half_life_days=1)
    entry = store.add(
        "Pinned fact", topic="core", pinned=True, created_at=now - timedelta(days=365)
    )

    weighted = store.get_weighted_memories(now=now)
    pinned_mem = next(m for m in weighted if m["id"] == entry.id)
    assert pinned_mem["weight"] == 1.0


def test_forget():
    store = make_store()
    entry = store.add("Temporary", topic="temp")
    assert len(store) == 1
    result = store.forget(entry.id)
    assert result is True
    assert len(store) == 0


def test_get_context_memories_top_k():
    store = make_store()
    for i in range(20):
        store.add(f"Memory {i}", topic=f"topic_{i}")
    memories = store.get_context_memories(top_k=5)
    assert len(memories) <= 5


def test_pin():
    store = make_store()
    entry = store.add("Important fact", topic="core")
    assert not entry.is_pinned
    store.pin(entry.id)
    assert entry.is_pinned


def test_forget_nonexistent_returns_false():
    store = make_store()
    result = store.forget("nonexistent-id")
    assert result is False


def test_all_entries():
    store = make_store()
    store.add("First", topic="a")
    store.add("Second", topic="b")
    entries = store.all_entries()
    assert len(entries) == 2


def test_repr():
    store = make_store()
    r = repr(store)
    assert "MemoryStore" in r
    assert "exponential" in r
