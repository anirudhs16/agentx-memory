# main.py
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import ResearchState
from agents import (
    memory_loader_node,
    searcher_node,
    synthesiser_node,
    critic_node,
    verdict_node,
    memory_saver_node,
    store
)

# ── Router ─────────────────────────────────────────────
def should_retry(state: ResearchState) -> str:
    if state["confidence"] < 70 and state["retry_count"] < 3:
        print(f"\n🔄 Confidence {state['confidence']}% - retrying ({state['retry_count']}/3)")
        return "retry"
    return "done"

# ── Build Graph ────────────────────────────────────────
def build_graph():
    # MemorySaver = short term (within a thread)
    # store = long term (across threads) — imported from agents.py
    checkpointer = MemorySaver()

    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("memory_loader", memory_loader_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("synthesiser", synthesiser_node)
    graph.add_node("critic", critic_node)
    graph.add_node("verdict", verdict_node)
    graph.add_node("memory_saver", memory_saver_node)

    # Sequential edges
    graph.add_edge(START, "memory_loader")
    graph.add_edge("memory_loader", "searcher")
    graph.add_edge("searcher", "synthesiser")
    graph.add_edge("synthesiser", "critic")
    graph.add_edge("critic", "verdict")

    # Conditional edge from verdict
    graph.add_conditional_edges(
        "verdict",
        should_retry,
        {
            "retry": "searcher",
            "done": "memory_saver"  # save before ending
        }
    )

    graph.add_edge("memory_saver", END)

    # Pass BOTH checkpointer and store
    return graph.compile(checkpointer=checkpointer, store=store)

# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    # Config ties the run to a specific thread
    # Same thread_id = MemorySaver remembers within session
    config = {"configurable": {"thread_id": "thread_001"}}

    # ── First run ──────────────────────────────────────
    print("=" * 50)
    print("RUN 1 — First time asking")
    print("=" * 50)

    result = app.invoke(
        ResearchState(
            query="What is LangGraph and why use it?",
            searcher_output="",
            synthesiser_output="",
            critic_output="",
            verdict_output="",
            confidence=0,
            retry_count=0,
            user_id="user_001",
            thread_id="thread_001",
            context=""
        ),
        config=config
    )

    print("\nVERDICT:", result["verdict_output"][:300])
    print(f"Confidence: {result['confidence']}%")

    # ── Second run ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("RUN 2 — Follow up question, same thread")
    print("=" * 50)

    result2 = app.invoke(
        ResearchState(
            query="How does LangGraph memory work specifically?",
            searcher_output="",
            synthesiser_output="",
            critic_output="",
            verdict_output="",
            confidence=0,
            retry_count=0,
            user_id="user_001",
            thread_id="thread_001",
            context=""
        ),
        config=config
    )

    print("\nVERDICT:", result2["verdict_output"][:300])
    print(f"Confidence: {result2['confidence']}%")
    print("\n✅ Notice: Run 2 had prior context from Run 1 loaded automatically")