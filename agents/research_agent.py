from typing import Dict

from tools import search_web
from .base_agent import BaseAgent

RESEARCH_SYSTEM_PROMPT = """You are an expert Research AI agent. Your role is to gather,
synthesize, and present accurate, relevant information on any topic.

You have access to a web search tool. Use it proactively to find current, specific information
rather than relying solely on your training data.

When researching a topic:
1. Break the topic into key questions to answer
2. Search for each major question separately for comprehensive coverage
3. Synthesize findings into a coherent, structured report
4. Cite which search queries produced which insights
5. Flag any conflicting information or areas of uncertainty

Your research reports should be:
- Factual and evidence-based
- Well-organized with clear sections
- Practical and actionable
- Honest about limitations and gaps

Output format:
## Research Summary: [Topic]

### Key Findings
[Bullet-point summary of the most important discoveries]

### Detailed Analysis
[Section-by-section breakdown organized by subtopic]

### Sources & Queries Used
[List the search queries you ran]

### Gaps & Uncertainties
[What remains unknown or needs further investigation]
"""

SEARCH_TOOL = {
    "name": "search_web",
    "description": (
        "Search the web for current information on a topic. "
        "Use specific, targeted queries for best results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string",
            }
        },
        "required": ["query"],
    },
}


class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="Researcher",
            role="Information gathering and synthesis",
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
        )

    def _handle_tool(self, tool_name: str, tool_input: Dict) -> str:
        if tool_name == "search_web":
            return search_web(tool_input.get("query", ""))
        return super()._handle_tool(tool_name, tool_input)
