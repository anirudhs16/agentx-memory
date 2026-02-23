# state.py
from typing import TypedDict

class ResearchState(TypedDict):
    # The current question
    query: str
    
    # Agent outputs
    searcher_output: str
    synthesiser_output: str
    critic_output: str
    verdict_output: str
    
    # Routing
    confidence: int
    retry_count: int
    
    # Memory fields — new
    user_id: str        # who is asking
    thread_id: str      # which conversation thread
    context: str        # prior research loaded from memory

# ```

# ---

# ## What's new — the three memory fields

# **`user_id`** — identifies who is using the system. Later when you build multi-user apps, each user has their own memory store.

# **`thread_id`** — identifies which conversation. Same user can have multiple research threads running simultaneously. This is how ChatGPT's conversation history works under the hood.

# **`context`** — this is the key one. Before the Searcher runs, a new memory loader node reads previous research for this thread and loads it here. The Searcher then gets both the new query AND previous context. That's what makes it feel like it remembers you.

# ---

# The flow will be:
# ```
# memory_loader → searcher → synthesiser → critic → verdict → memory_saver
#                                                        ↓
#                                               (conditional retry)