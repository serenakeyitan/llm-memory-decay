"""
OpenAI integration example.

Shows how to use DecayingMemoryManager to inject decayed memories
into an OpenAI chat completion request.
"""
from datetime import datetime, timedelta

from llm_memory_decay.decay import DecayStrategy
from llm_memory_decay.integrations.openai import DecayingMemoryManager

# Simulate an OpenAI memory dump (what their system would give you)
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
    {
        "content": "User is learning Rust",
        "topic": "learning",
        "importance": 1.5,
        "created_at": (datetime.utcnow() - timedelta(days=14)).isoformat(),
    },
]

# Initialize with 30-day half-life
manager = DecayingMemoryManager(
    decay_strategy=DecayStrategy.EXPONENTIAL,
    half_life_days=30.0,
    top_k=5,
)

# Ingest the raw memories
manager.ingest_openai_memories(raw_memories)

# Build a message list with decayed memories injected
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Can you help me debug this Python script?"},
]

enriched = manager.inject_into_messages(messages)
print("=== Enriched Messages ===")
for msg in enriched:
    print(f"\n[{msg['role'].upper()}]")
    print(msg["content"])

print("\n=== Memory Manager Info ===")
print(repr(manager))

# To actually send to OpenAI (requires openai package + API key):
# from openai import OpenAI
# client = OpenAI()
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=enriched,
# )
# print(response.choices[0].message.content)
