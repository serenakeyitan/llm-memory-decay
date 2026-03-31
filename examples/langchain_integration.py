"""
LangChain integration example.

Requires: pip install "llm-memory-decay[langchain]"

Shows how to use DecayingConversationMemory as a drop-in replacement
for LangChain's built-in ConversationBufferMemory.
"""
from llm_memory_decay.integrations.langchain import LANGCHAIN_AVAILABLE, DecayingConversationMemory

if not LANGCHAIN_AVAILABLE:
    print("LangChain is not installed. Install it with: pip install 'llm-memory-decay[langchain]'")
    print("\nShowing standalone usage instead:\n")

    # Standalone usage (no LangChain required)
    memory = DecayingConversationMemory(half_life_days=14, top_k=8)
    memory.add_memory("User is a backend engineer", topic="profession", importance=2.0)
    memory.add_memory("User prefers Rust over C++", topic="preferences", importance=1.5)
    memory.add_memory("User asked about Python packaging once", topic="packaging", importance=1.0)

    print("Memory context:")
    print(memory.get_memory_context())
    print(f"\nDecay store size: {len(memory.decay_store)} entries")
else:
    # Full LangChain usage
    try:
        from langchain.chains import ConversationChain
        from langchain_openai import ChatOpenAI

        memory = DecayingConversationMemory(half_life_days=14, top_k=8)
        memory.add_memory("User is a backend engineer", topic="profession", importance=2.0)
        memory.add_memory("User prefers Rust over C++", topic="preferences", importance=1.5)

        # Requires OPENAI_API_KEY in environment
        chain = ConversationChain(
            llm=ChatOpenAI(model="gpt-4o-mini"),
            memory=memory,
        )

        response = chain.predict(
            input="What language should I use for a new high-performance system service?"
        )
        print(response)

    except ImportError as e:
        print(f"Additional dependency missing: {e}")
        print("Try: pip install langchain-openai")
