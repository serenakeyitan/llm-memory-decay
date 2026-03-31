"""
LangChain memory adapter for llm-memory-decay.

Usage:
    from llm_memory_decay.integrations.langchain import DecayingConversationMemory

    memory = DecayingConversationMemory(half_life_days=14)
    chain = ConversationChain(llm=llm, memory=memory)
"""
from __future__ import annotations

try:
    from langchain.memory.chat_memory import BaseChatMemory

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from llm_memory_decay.decay import DecayStrategy
from llm_memory_decay.filters import AntiObsessionConfig
from llm_memory_decay.formatters import format_for_system_prompt
from llm_memory_decay.memory import MemoryStore


class DecayingMemoryMixin:
    """
    Mixin that adds memory decay to any LangChain memory class.
    Can be used standalone or as a mixin.
    """

    def __init__(
        self,
        decay_strategy: DecayStrategy = DecayStrategy.EXPONENTIAL,
        half_life_days: float = 30.0,
        top_k: int = 10,
        anti_obsession: AntiObsessionConfig | None = None,
        **kwargs,
    ) -> None:
        self._decay_store = MemoryStore(
            decay_strategy=decay_strategy,
            half_life_days=half_life_days,
            anti_obsession=anti_obsession,
        )
        self._top_k = top_k
        if LANGCHAIN_AVAILABLE:
            super().__init__(**kwargs)

    def add_memory(
        self, content: str, topic: str = "", importance: float = 1.0
    ) -> None:
        self._decay_store.add(content=content, topic=topic, importance=importance)

    def get_memory_context(self) -> str:
        return format_for_system_prompt(self._decay_store, top_k=self._top_k)

    @property
    def decay_store(self) -> MemoryStore:
        return self._decay_store


if LANGCHAIN_AVAILABLE:

    class DecayingConversationMemory(DecayingMemoryMixin, BaseChatMemory):
        """Drop-in LangChain memory with principled decay."""

        memory_key: str = "history"

        @property
        def memory_variables(self) -> list[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: dict) -> dict:
            return {self.memory_key: self.get_memory_context()}

        def save_context(self, inputs: dict, outputs: dict) -> None:
            human_input = inputs.get("input", "")
            ai_output = outputs.get("response", outputs.get("output", ""))
            if human_input:
                self._decay_store.add(
                    content=f"User said: {human_input}", topic="conversation"
                )
            if ai_output:
                self._decay_store.add(
                    content=f"Assistant said: {ai_output}", topic="conversation"
                )

else:

    class DecayingConversationMemory(DecayingMemoryMixin):  # type: ignore
        """Stub when LangChain is not installed. Install langchain to use."""

        pass
