# agentx-memory 🧠💾

> Multi-agent research system with short-term and long-term memory.

Builds on [agentx-graph](https://github.com/anirudhs16/agentx-graph) by adding two memory layers — a checkpointer that resumes paused runs, and a persistent store that remembers context across separate sessions.

![Stack](https://img.shields.io/badge/stack-LangGraph%20%2B%20Groq%20%2B%20Python-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## What's New vs agentx-graph

| Feature | agentx-graph | agentx-memory |
|---|---|---|
| Sequential flow | ✅ | ✅ |
| Conditional retry | ✅ | ✅ |
| Short-term memory | ❌ | ✅ MemorySaver |
| Long-term memory | ❌ | ✅ InMemoryStore |
| Cross-session context | ❌ | ✅ |

---

## Memory Architecture

```
Session 1:
memory_loader (no prior context) → searcher → synthesiser → critic → verdict → memory_saver
                                                                           ↓
                                                                   saves to store

Session 2 (same thread):
memory_loader (loads Session 1 summary) → searcher (context-aware) → ... → memory_saver
```

### Two Memory Layers

**Short-term — MemorySaver (checkpointer)**
- Tied to `thread_id` in config
- Persists full state within a session
- Allows pausing and resuming a run mid-execution
- Cleared when process exits

**Long-term — InMemoryStore**
- Tied to `(user_id, namespace)`
- Saves research summaries across separate invocations
- In production: swap `InMemoryStore` for `PostgresStore` — zero agent code changes needed

---

## Graph Structure

```
START
  ↓
memory_loader      ← loads prior research from store
  ↓
searcher           ← uses prior context if available
  ↓
synthesiser
  ↓
critic
  ↓
verdict
  ↓ (confidence < 70 AND retries < 3)
  ↓ retry → searcher
  ↓ done
memory_saver       ← saves summary to store
  ↓
END
```

---

## Tech Stack

- Python 3.10+
- LangGraph — graph orchestration + memory
- LangChain Groq — LLM integration
- Groq API — Llama 3.3 70B / 3.1 8B (free tier)

---

## Getting Started

### Prerequisites
- Python 3.10+
- Free [Groq API key](https://console.groq.com)

### Setup

```bash
git clone https://github.com/anirudhs16/agentx-memory.git
cd agentx-memory

python3 -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt

echo "GROQ_API_KEY=your_key_here" > .env
```

### Run

```bash
python main.py
```

---

## Project Structure

```
agentx-memory/
├── main.py       # graph definition + memory config
├── agents.py     # nodes including memory_loader + memory_saver
├── state.py      # shared state with memory fields
└── .env          # GROQ_API_KEY (git ignored)
```

---

## Key Learning — Production Memory Pattern

```python
# Development
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# Production — same API, just swap the store
from langgraph.store.postgres import PostgresStore
store = PostgresStore(connection_string="postgresql://...")
```

Agent code stays identical. Only the store backend changes.

---

## Learning Progression

```
agentx          ← sequential pipeline
agentx-graph    ← conditional graph + retry loop
agentx-memory   ← short + long term memory (you are here)
agentx-tools    ← tool calling + function execution (next)
```

---

## License

MIT