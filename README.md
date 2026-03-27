# 🧠 llm-memory-decay

**Smart Forgetting for LLM Personalization**

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![CI](https://github.com/serenakeyitan/llm-memory-decay/workflows/CI/badge.svg)
![PyPI](https://img.shields.io/pypi/v/llm-memory-decay.svg)

---

> *"A single question from 2 months ago about some topic can keep coming up as some kind of a deep interest of mine with undue mentions in perpetuity."*
>
> — **Andrej Karpathy** on built-in LLM memory systems

---

## The Problem

Modern LLMs (ChatGPT, Claude, Gemini) all offer some form of persistent memory — but their implementations share a critical flaw: **they don't forget**.

A question you asked once, two months ago, might get weighted as a core personality trait forever. Ask about sourdough bread once? Congratulations — you're now a "baking enthusiast" in perpetuity. Your AI assistant will keep bringing it up, suggesting recipes, referencing your "love of baking" long after you've moved on.

This is the **memory obsession problem**:

- **Recency blindness**: A memory from 6 months ago carries the same weight as one from yesterday
- **One-time query inflation**: A single casual question gets treated as a persistent interest
- **Topic dominance**: One salient topic can crowd out dozens of more relevant, recent ones
- **No principled decay**: Without forgetting curves, memory just accumulates noise

---

## The Solution

`llm-memory-decay` implements **principled memory decay** — the same forgetting curves used in spaced repetition systems (Anki, SuperMemo) but applied to LLM personalization context.

| Feature | Built-in LLM Memory | llm-memory-decay |
|---|---|---|
| Recency weighting | ❌ Flat / none | ✅ Exponential, linear, step |
| One-time query handling | ❌ Treated as strong signal | ✅ Anti-obsession penalty |
| Topic dominance control | ❌ No cap | ✅ Per-topic surface rate cap |
| User-controllable decay | ❌ Black box | ✅ Fully configurable |
| Pinned memories | ❌ All-or-nothing | ✅ Pin specific facts |
| Open source | ❌ Proprietary | ✅ MIT License |
| Framework integration | ❌ Vendor lock-in | ✅ LangChain + OpenAI adapters |

---

## Installation

```bash
pip install llm-memory-decay
```

With LangChain support:
```bash
pip install "llm-memory-decay[langchain]"
```

With OpenAI support:
```bash
pip install "llm-memory-decay[openai]"
```

All extras:
```bash
pip install "llm-memory-decay[all]"
```

---

## Quick Start

```python
from datetime import datetime, timedelta
from llm_memory_decay import MemoryStore, DecayStrategy
from llm_memory_decay.formatters import format_for_system_prompt

# Create a store with 30-day half-life exponential decay
store = MemoryStore(
    decay_strategy=DecayStrategy.EXPONENTIAL,
    half_life_days=30.0,
)

now = datetime.utcnow()

# Add memories — older ones naturally fade
store.add("User is a Python developer", topic="profession", importance=2.0,
          created_at=now - timedelta(days=5))
store.add("User asked about TypeScript once", topic="typescript", importance=1.0,
          created_at=now - timedelta(days=90))
store.add("User loves hiking", topic="hobbies", importance=1.5,
          created_at=now - timedelta(days=10))
store.add("User asked about tax filing", topic="finance", importance=1.0,
          created_at=now - timedelta(days=180))

# See what survives decay
for mem in store.get_context_memories(top_k=5, now=now):
    print(f"[{mem['weight']:.2f}] {mem['content']}")
# [1.89] User is a Python developer
# [1.03] User loves hiking
# [0.05] User asked about TypeScript once   ← fading fast
# (tax filing dropped off entirely)

# Inject into your system prompt
print(format_for_system_prompt(store, top_k=3, now=now))
# ## User Memory Context
#
# - User is a Python developer (relevance: 189%)
# - User loves hiking (relevance: 103%)
# - User asked about TypeScript once (relevance: 5%)
```

---

## Decay Strategies

### Exponential Decay (default)
Mathematically principled. Weight halves every `half_life_days`.

```python
store = MemoryStore(
    decay_strategy=DecayStrategy.EXPONENTIAL,
    half_life_days=30.0,  # weight = 0.5 at 30 days, 0.25 at 60 days
)
```

Best for: continuous personalization where recent context matters most.

### Linear Decay
Weight drops linearly to zero at `max_age_days`.

```python
store = MemoryStore(
    decay_strategy=DecayStrategy.LINEAR,
    max_age_days=90.0,  # weight = 1.0 at day 0, 0.0 at day 90
)
```

Best for: hard cutoffs — "only remember the last 3 months."

### Step Decay
Stepped buckets: recent (≤7d) = 1.0, medium (≤30d) = 0.5, old (≤90d) = 0.1, ancient = 0.

```python
store = MemoryStore(decay_strategy=DecayStrategy.STEP)
```

Best for: simple systems that want discrete tiers without continuous math.

### No Decay
All memories carry full weight. Equivalent to naive LLM memory.

```python
store = MemoryStore(decay_strategy=DecayStrategy.NONE)
```

Best for: testing, debugging, or short-lived sessions where decay doesn't apply.

---

## Anti-Obsession Filter

The anti-obsession filter is what makes this library genuinely better than built-in solutions.

```python
from llm_memory_decay.filters import AntiObsessionConfig

config = AntiObsessionConfig(
    max_surface_rate=0.3,          # no topic can take >30% of context slots
    min_effective_weight=0.05,     # prune memories below 5% relevance
    one_time_query_threshold=1,    # topics seen only once are "one-time queries"
    one_time_decay_multiplier=0.3, # one-time queries get 70% weight reduction
)

store = MemoryStore(anti_obsession=config)
```

**How it works:**

1. **One-time query detection**: If a topic appears in memory only once (frequency=1), it's flagged as a casual question, not a deep interest. Its weight is multiplied by `one_time_decay_multiplier` (default 0.3).

2. **Minimum weight pruning**: Memories below `min_effective_weight` are removed entirely — they'd add noise, not signal.

3. **Per-topic dominance cap**: No single topic can occupy more than `max_surface_rate` of your context slots. If you have 20 context slots and `max_surface_rate=0.3`, any single topic is capped at 6 slots, leaving room for diversity.

---

## LangChain Integration

```python
from llm_memory_decay.integrations.langchain import DecayingConversationMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Drop-in replacement for ConversationBufferMemory
memory = DecayingConversationMemory(
    half_life_days=14,   # aggressive decay — 2-week half-life
    top_k=8,
)

# Add explicit facts
memory.add_memory("User is a backend engineer", topic="profession", importance=2.0)
memory.add_memory("User prefers Rust over C++", topic="preferences", importance=1.5)

chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-4o"),
    memory=memory,
)

response = chain.predict(input="What language should I use for a new system service?")
# The memory context is automatically injected — and decays over time
```

---

## OpenAI Integration

```python
from datetime import datetime, timedelta
from llm_memory_decay.integrations.openai import DecayingMemoryManager

# Simulate memories exported from OpenAI / your own system
raw_memories = [
    {
        "content": "User is a machine learning engineer",
        "topic": "profession",
        "importance": 2.0,
        "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
    },
    {
        "content": "User asked about sourdough bread once",
        "topic": "cooking",
        "importance": 1.0,
        "created_at": (datetime.utcnow() - timedelta(days=120)).isoformat(),
    },
    {
        "content": "User prefers concise answers",
        "topic": "preferences",
        "importance": 2.0,
        "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
    },
]

manager = DecayingMemoryManager(half_life_days=30.0, top_k=5)
manager.ingest_openai_memories(raw_memories)

# Inject into your messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you help me debug this Python script?"},
]

enriched = manager.inject_into_messages(messages)
# The first system message now includes the decayed memory context
# The sourdough memory is penalized (one-time + 120 days old) and likely drops off
# The profession and preferences memories survive with high weights
```

---

## Pinning Important Memories

Some facts should never fade — user's name, critical preferences, accessibility needs:

```python
store = MemoryStore()

# Pinned memories always have weight=1.0, regardless of age
entry = store.add("User is deaf — always provide text, never audio-only",
                  topic="accessibility", pinned=True)

# Or pin after the fact
entry2 = store.add("User's name is Alex", topic="identity")
store.pin(entry2.id)

# Forget something explicitly
store.forget(some_entry.id)
```

---

## API Reference

### `MemoryStore`

The core class. Thread-safe for read access; add a lock if writing from multiple threads.

| Method | Description |
|---|---|
| `add(content, topic, importance, created_at, metadata, pinned)` | Add a memory entry |
| `get_weighted_memories(now)` | All memories with computed weights, after filtering |
| `get_context_memories(top_k, now)` | Top-k memories sorted by weight |
| `forget(memory_id)` | Remove a memory by ID |
| `pin(memory_id)` | Pin a memory so it never decays |
| `all_entries()` | Raw list of all `MemoryEntry` objects |

### `DecayStrategy`

Enum: `EXPONENTIAL`, `LINEAR`, `STEP`, `NONE`

### `AntiObsessionConfig`

| Field | Default | Description |
|---|---|---|
| `max_surface_rate` | 0.3 | Max fraction of context slots per topic |
| `min_effective_weight` | 0.05 | Prune memories below this weight |
| `one_time_query_threshold` | 1 | Frequency threshold for "one-time query" classification |
| `one_time_decay_multiplier` | 0.3 | Weight penalty for one-time queries |

### `ImportanceScorer`

| Method | Description |
|---|---|
| `record_mention(topic, at, importance)` | Track a topic mention |
| `get_frequency_weight(topic, max_frequency)` | Normalized [0.1, 1.0] frequency score |
| `pin_topic(topic)` | Mark a topic as always-relevant |
| `is_pinned(topic)` | Check if a topic is pinned |

### Formatters

```python
from llm_memory_decay.formatters import format_for_system_prompt, format_as_list

# Returns a markdown-formatted string for system prompt injection
text = format_for_system_prompt(store, top_k=10, header="## My Context")

# Returns a plain list of content strings
memories_list = format_as_list(store, top_k=10)
```

---

## Why Not Just Use Built-in Memory?

The built-in memory systems from OpenAI, Anthropic, and Google are:

- **Opaque**: You can't inspect, tune, or reason about what's remembered and why
- **Non-decaying**: Memories persist indefinitely without recency weighting
- **Vendor-locked**: Your user's memory lives in their infrastructure
- **One-size-fits-all**: No way to adjust decay rates per use case

`llm-memory-decay` gives you full control, full transparency, and framework independence.

---

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Write tests for new functionality
4. Ensure `ruff check` and `pytest` pass
5. Open a pull request

Bug reports and feature requests go in [Issues](https://github.com/serenakeyitan/llm-memory-decay/issues).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

Copyright (c) 2025 serenakeyitan
