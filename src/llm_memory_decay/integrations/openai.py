"""
OpenAI memory helper for llm-memory-decay.

Usage:
    from llm_memory_decay.integrations.openai import DecayingMemoryManager

    manager = DecayingMemoryManager()
    manager.ingest_openai_memories(raw_memories)
    system_prompt = manager.build_system_prompt_section()
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from llm_memory_decay.memory import MemoryStore
from llm_memory_decay.decay import DecayStrategy
from llm_memory_decay.filters import AntiObsessionConfig
from llm_memory_decay.formatters import format_for_system_prompt


class DecayingMemoryManager:
    """
    Manages decaying memories for OpenAI-style conversation threads.

    Works with OpenAI chat format: list of {role, content} dicts.
    Extracts key facts and applies decay before injecting into system prompt.
    """

    def __init__(
        self,
        decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL,
        half_life_days: float = 30.0,
        top_k: int = 10,
        anti_obsession: AntiObsessionConfig | None = None,
    ) -> None:
        self.store = MemoryStore(
            decay_strategy=decay_strategy,
            half_life_days=half_life_days,
            anti_obsession=anti_obsession or AntiObsessionConfig(),
        )
        self.top_k = top_k

    def add_memory(
        self,
        content: str,
        topic: str = "general",
        importance: float = 1.0,
        created_at: datetime | None = None,
        pinned: bool = False,
    ) -> None:
        """Manually add a memory fact."""
        self.store.add(
            content=content,
            topic=topic,
            importance=importance,
            created_at=created_at,
            pinned=pinned,
        )

    def ingest_openai_memories(self, memories: list[dict[str, Any]]) -> None:
        """
        Ingest memories from OpenAI memory format.

        Expected format: [{"content": "...", "created_at": "...", "topic": "..."}]
        """
        for mem in memories:
            content = mem.get("content", "")
            topic = mem.get("topic", "general")
            importance = float(mem.get("importance", 1.0))
            created_at_str = mem.get("created_at")
            created_at = None
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                    # Strip timezone for comparison
                    created_at = created_at.replace(tzinfo=None)
                except ValueError:
                    pass
            if content:
                self.store.add(
                    content=content,
                    topic=topic,
                    importance=importance,
                    created_at=created_at,
                )

    def build_system_prompt_section(
        self, header: str = "## Personalization Context"
    ) -> str:
        """Build the memory section to prepend to your system prompt."""
        return format_for_system_prompt(self.store, top_k=self.top_k, header=header)

    def inject_into_messages(
        self,
        messages: list[dict],
        system_header: str = "## Personalization Context",
    ) -> list[dict]:
        """
        Inject decayed memories into an OpenAI messages list.

        Prepends memory context to the first system message, or inserts one.
        """
        memory_section = self.build_system_prompt_section(header=system_header)
        if not memory_section:
            return messages

        result = list(messages)
        for i, msg in enumerate(result):
            if msg.get("role") == "system":
                result[i] = {**msg, "content": memory_section + "\n\n" + msg["content"]}
                return result

        # No system message found — prepend one
        return [{"role": "system", "content": memory_section}] + result

    def __repr__(self) -> str:
        return (
            f"DecayingMemoryManager(entries={len(self.store)}, "
            f"strategy={self.store.decay_strategy})"
        )
