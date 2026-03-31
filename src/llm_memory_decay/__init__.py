"""
llm-memory-decay: Smart forgetting for LLM personalization.

Implements principled memory decay to prevent stale memories from
dominating LLM context — inspired by Karpathy's observation about
LLM memory "trying too hard."
"""

from .decay import DecayStrategy, compute_decay_weight
from .filters import AntiObsessionConfig, AntiObsessionFilter
from .formatters import format_as_list, format_for_system_prompt
from .memory import MemoryEntry, MemoryStore
from .scoring import ImportanceScorer, TopicScore

__version__ = "0.1.0"
__all__ = [
    "MemoryStore",
    "MemoryEntry",
    "DecayStrategy",
    "compute_decay_weight",
    "ImportanceScorer",
    "TopicScore",
    "AntiObsessionFilter",
    "AntiObsessionConfig",
    "format_for_system_prompt",
    "format_as_list",
]
