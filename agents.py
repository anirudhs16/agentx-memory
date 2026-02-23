# agents.py
from langchain_groq import ChatGroq
from langgraph.store.memory import InMemoryStore
from state import ResearchState

# Initialise models
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
llm_fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Initialise long-term store
# This persists across separate invocations
store = InMemoryStore()

# ── Memory Loader ──────────────────────────────────────
def memory_loader_node(state: ResearchState) -> ResearchState:
    """Runs first. Loads any prior research for this thread."""
    
    user_id = state.get("user_id", "default")
    thread_id = state.get("thread_id", "default")
    
    # Namespace isolates each user's memory
    namespace = (user_id, "research")
    
    # Try to load existing memory for this thread
    existing = store.get(namespace, thread_id)
    
    if existing:
        context = existing.value.get("summary", "")
        print(f"📚 Memory loaded for thread: {thread_id}")
        print(f"Previous context: {context[:100]}...")
    else:
        context = ""
        print(f"🆕 No prior memory for thread: {thread_id}")
    
    return {"context": context}

# ── Searcher Node ──────────────────────────────────────
def searcher_node(state: ResearchState) -> ResearchState:
    # Include prior context if it exists
    prior = f"\nPrior research context:\n{state['context']}" \
            if state['context'] else ""
    
    response = llm_fast.invoke([
        ("system", f"""You are a Searcher agent.
         Find raw facts and evidence only.
         No opinions. No conclusions. Just evidence.
         {prior}"""),
        ("human", f"Research this: {state['query']}")
    ])
    
    return {"searcher_output": response.content}

# ── Synthesiser Node ───────────────────────────────────
def synthesiser_node(state: ResearchState) -> ResearchState:
    response = llm.invoke([
        ("system", """You are a Synthesiser agent.
         Build a clear structured answer from the evidence.
         Be decisive. Draw conclusions."""),
        ("human", f"""Question: {state['query']}
         Evidence: {state['searcher_output']}""")
    ])
    
    return {"synthesiser_output": response.content}

# ── Critic Node ────────────────────────────────────────
def critic_node(state: ResearchState) -> ResearchState:
    response = llm_fast.invoke([
        ("system", """You are a Critic agent.
         Aggressively challenge the answer below.
         Find holes, assumptions, missing evidence.
         Be harsh but fair."""),
        ("human", f"Answer to critique: {state['synthesiser_output']}")
    ])
    
    return {"critic_output": response.content}

# ── Verdict Node ───────────────────────────────────────
def verdict_node(state: ResearchState) -> ResearchState:
    response = llm.invoke([
        ("system", """You are a Supervisor delivering a final verdict.
        Structure your response exactly like this:
        
        ANSWER: [Direct answer to the original query in 2-3 sentences]
        
        REASONING: [What the agents got right and wrong]
        
        CONFIDENCE: [number between 0 and 100]
        Nothing after the CONFIDENCE line."""),
        ("human", f"""Query: {state['query']}
         Searcher found: {state['searcher_output']}
         Synthesiser concluded: {state['synthesiser_output']}
         Critic challenged: {state['critic_output']}
         
         Deliver your final verdict.""")
    ])
    
    content = response.content
    confidence = 75  # default

    for line in content.strip().split("\n"):
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line.replace("CONFIDENCE:", "").strip())
            except:
                pass

    return {
        "verdict_output": content,
        "confidence": confidence,
        "retry_count": state.get("retry_count", 0) + 1
    }

# ── Memory Saver ───────────────────────────────────────
def memory_saver_node(state: ResearchState) -> ResearchState:
    """Runs last. Saves research summary to long-term store."""
    
    user_id = state.get("user_id", "default")
    thread_id = state.get("thread_id", "default")
    namespace = (user_id, "research")

    # Build a summary to store
    summary = f"""
Query: {state['query']}
Verdict: {state['verdict_output'][:500]}
Confidence: {state['confidence']}%
    """.strip()

    # Save to store under this thread
    store.put(namespace, thread_id, {"summary": summary})
    
    print(f"\n💾 Memory saved for thread: {thread_id}")
    
    return {}