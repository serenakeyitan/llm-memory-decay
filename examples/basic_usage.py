"""Basic usage of llm-memory-decay."""
from datetime import datetime, timedelta

from llm_memory_decay import MemoryStore, DecayStrategy
from llm_memory_decay.formatters import format_for_system_prompt

# Create a store with 30-day half-life exponential decay
store = MemoryStore(
    decay_strategy=DecayStrategy.EXPONENTIAL,
    half_life_days=30.0,
)

now = datetime.utcnow()

# Add some memories with different ages
store.add(
    "User is a Python developer",
    topic="profession",
    importance=2.0,
    created_at=now - timedelta(days=5),
)
store.add(
    "User asked about TypeScript once",
    topic="typescript",
    importance=1.0,
    created_at=now - timedelta(days=90),
)
store.add(
    "User loves hiking",
    topic="hobbies",
    importance=1.5,
    created_at=now - timedelta(days=10),
)
store.add(
    "User asked about tax filing",
    topic="finance",
    importance=1.0,
    created_at=now - timedelta(days=180),
)

# See weighted memories
print("=== Weighted Memories ===")
for mem in store.get_context_memories(top_k=10, now=now):
    print(f"  [{mem['weight']:.2f}] {mem['content']}")

# Format for system prompt
print("\n=== System Prompt Section ===")
print(format_for_system_prompt(store, top_k=5, now=now))
