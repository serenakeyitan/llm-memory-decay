from datetime import datetime

from .memory import MemoryStore


def format_for_system_prompt(
    store: MemoryStore,
    top_k: int = 10,
    header: str = "## User Memory Context",
    now: datetime | None = None,
) -> str:
    """Format top memories as a system prompt section."""
    memories = store.get_context_memories(top_k=top_k, now=now)
    if not memories:
        return ""

    lines = [header, ""]
    for mem in memories:
        score_pct = int(mem["weight"] * 100)
        lines.append(f"- {mem['content']} (relevance: {score_pct}%)")

    return "\n".join(lines)


def format_as_list(
    store: MemoryStore, top_k: int = 10, now: datetime | None = None
) -> list[str]:
    """Return memories as a plain list of strings."""
    memories = store.get_context_memories(top_k=top_k, now=now)
    return [m["content"] for m in memories]
