# Multi-Agent AI Collaboration System

A Python-based agent orchestration system using the Anthropic Claude API. A **Manager Agent** coordinates a team of specialized sub-agents via Claude's tool-use loop to accomplish complex tasks end-to-end.

## Architecture

```text
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Manager Agent                 │
│  (claude-opus-4-7 + adaptive thinking)  │
│  Orchestrates via tool-use loop         │
└────┬──────┬──────┬──────┬──────────────┘
     │      │      │      │
     ▼      ▼      ▼      ▼
 Planner Research Coder Reviewer
 Agent   Agent   Agent  Agent
```

**Agents:**

| Agent | Role |
| --- | --- |
| Manager | Central orchestrator — delegates, iterates, synthesizes |
| Planner | Decomposes tasks into phased execution plans |
| Researcher | Gathers information via web search |
| Coder | Writes complete, production-ready code |
| Reviewer | Reviews code for quality, bugs, and security |

**Key features:**

- Native Claude tool-use agentic loop (no LangGraph/CrewAI dependency)
- Adaptive thinking on the Manager Agent for complex reasoning
- Thread-safe shared memory for cross-agent context passing
- Conflict resolution: Reviewer → Coder → Reviewer revision cycles
- Prompt caching on all system prompts to reduce token costs

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
ANTHROPIC_API_KEY=your_key_here
```

## Usage

**Interactive mode:**

```bash
python main.py
```

**Single query:**

```bash
python main.py --query "Build a Python rate limiter using the token bucket algorithm"
```

**Quiet mode (no agent activity logs):**

```bash
python main.py --query "..." --quiet
```

## Project Structure

```text
├── main.py                    # Entry point (CLI + interactive REPL)
├── requirements.txt
├── agents/
│   ├── base_agent.py          # BaseAgent with run() and run_with_tools()
│   ├── manager_agent.py       # Orchestrator — owns the tool-use loop
│   ├── planner_agent.py
│   ├── research_agent.py      # Uses search_web tool
│   ├── coder_agent.py
│   └── reviewer_agent.py
├── core/
│   └── shared_memory.py       # Thread-safe key-value store
└── tools/
    └── search_tool.py         # Mock search (swap in Tavily/SerpAPI)
```

## Replacing Mock Search

In [tools/search_tool.py](tools/search_tool.py), replace `search_web()` with a real API call:

```python
from tavily import TavilyClient

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    results = client.search(query)
    return "\n".join(r["content"] for r in results["results"][:3])
```
