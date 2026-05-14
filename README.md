# Multi-Agent AI Collaboration System

A Python-based agent orchestration system powered by **Groq**. A **Manager Agent** coordinates a team of specialized sub-agents via a tool-use loop to accomplish complex tasks end-to-end.

## Architecture

```text
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              Manager Agent                  │
│  (llama-3.3-70b-versatile via Groq API)     │
│  Orchestrates via tool-use loop             │
└────┬──────┬──────┬──────┬───────────────────┘
     │      │      │      │
     ▼      ▼      ▼      ▼
 Planner Research Coder Reviewer
 Agent   Agent   Agent  Agent
 (8b)    (8b)    (8b)   (8b)
```

**Agents:**

| Agent | Model | Role |
| --- | --- | --- |
| Manager | `llama-3.3-70b-versatile` | Central orchestrator — delegates, iterates, synthesizes |
| Planner | `llama-3.1-8b-instant` | Decomposes tasks into phased execution plans |
| Researcher | `llama-3.1-8b-instant` | Gathers information via web search |
| Coder | `llama-3.1-8b-instant` | Writes complete, production-ready code |
| Reviewer | `llama-3.1-8b-instant` | Reviews code for quality, bugs, and security |

**Key features:**

- All agents call Groq API independently — 5–8 API calls per user query
- Thread-safe shared memory for cross-agent context passing
- Conflict resolution: Reviewer → Coder → Reviewer revision cycles (automatic)
- Malformed tool-call recovery: handles Llama's native `<function=...>` fallback format
- Rate limit retry with exponential backoff (free tier: 100K tokens/day)
- Rolling context window to prevent token overflow on long orchestrations

## Setup

**1. Install dependencies (local):**

```bash
pip install -r requirements.txt
```

**2. Create a `.env` file:**

```text
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com/keys](https://console.groq.com/keys).

**3. Build Docker image (optional):**

```bash
docker-compose build
```

## Usage

**Local — interactive REPL:**

```bash
python main.py
```

**Local — single query:**

```bash
python main.py --query "Build a Python rate limiter using the token bucket algorithm"
```

**Local — quiet mode:**

```bash
python main.py --query "..." --quiet
```

**Docker — interactive REPL:**

```bash
docker-compose run --rm agent
```

**Docker — single query:**

```bash
docker-compose run --rm agent --query "Build a Python rate limiter"
```

## Example Flow

```text
You: Build a Python function that checks if a number is prime

[Manager] → call_planner       Planner creates phased execution plan
[Manager] → call_researcher    Researcher gathers algorithm info
[Manager] → store_memory       Saves research findings to shared memory
[Manager] → call_coder         Coder writes the implementation
[Manager] → call_reviewer      Reviewer finds issues, requests changes
[Manager] → call_coder         Coder revises based on feedback
[Manager] → call_reviewer      Reviewer approves
[Manager] → Final response delivered
```

## Project Structure

```text
├── main.py                    # Entry point (CLI + interactive REPL)
├── requirements.txt           # groq, python-dotenv
├── Dockerfile
├── docker-compose.yml
├── agents/
│   ├── base_agent.py          # BaseAgent — Groq client, run() and run_with_tools()
│   ├── manager_agent.py       # Orchestrator — tool-use loop, recovery, rate limit retry
│   ├── planner_agent.py
│   ├── research_agent.py      # Uses search_web tool
│   ├── coder_agent.py
│   └── reviewer_agent.py
├── core/
│   └── shared_memory.py       # Thread-safe key-value store
└── tools/
    └── search_tool.py         # Mock search (swap in Tavily/SerpAPI)
```

## Groq Free Tier Limits

| Limit | Value |
| --- | --- |
| Tokens per minute (TPM) | 12,000 |
| Tokens per day (TPD) | 100,000 |
| Reset | Every 24 hours |

A typical query uses ~8,000–15,000 tokens across all agent calls.
Upgrade to Dev Tier at [console.groq.com/settings/billing](https://console.groq.com/settings/billing) for higher limits.

## Replacing Mock Search

In [tools/search_tool.py](tools/search_tool.py), replace `search_web()` with a real API call:

```python
from tavily import TavilyClient

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str) -> str:
    results = client.search(query)
    return "\n".join(r["content"] for r in results["results"][:3])
```
