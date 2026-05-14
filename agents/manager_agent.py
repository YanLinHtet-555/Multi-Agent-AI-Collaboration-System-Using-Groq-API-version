import anthropic

from core import SharedMemory
from .planner_agent import PlannerAgent
from .research_agent import ResearchAgent
from .coder_agent import CoderAgent
from .reviewer_agent import ReviewerAgent

MANAGER_SYSTEM_PROMPT = """You are the Manager Agent — the central orchestrator of a
multi-agent AI collaboration system. You coordinate a team of specialized agents to
accomplish complex user goals.

## Your Team
- **Planner**: Decomposes tasks into structured, phased execution plans
- **Researcher**: Gathers and synthesizes information using web search
- **Coder**: Writes complete, production-ready code implementations
- **Reviewer**: Reviews code and plans for quality, correctness, and security

## Your Responsibilities
1. **Understand** the user's goal fully before delegating
2. **Plan** the work by calling the Planner first for complex tasks
3. **Research** unknowns before asking the Coder to implement
4. **Delegate** specific, well-scoped tasks to the right agent
5. **Review** all code output via the Reviewer before delivering it
6. **Iterate** — if the Reviewer requests changes, call the Coder again with the feedback
7. **Store** important intermediate results in shared memory for cross-agent reuse
8. **Synthesize** all agent outputs into a coherent final response

## Decision Rules
- Always call Planner first for multi-step tasks
- Always call Reviewer after Coder produces code
- If Reviewer verdict is CHANGES REQUESTED, call Coder again with reviewer feedback,
  then call Reviewer again (max 3 revision cycles)
- Use store_memory to save research findings so Coder can reference them
- Use read_memory to retrieve context before delegating tasks
- When all steps are complete, synthesize outputs and deliver a final response

## Communication Style
- Be decisive — pick the right agent and give them a clear, scoped task
- Provide relevant context from memory when calling agents
- Your final response to the user should be comprehensive and well-structured
"""

MANAGER_TOOLS = [
    {
        "name": "call_planner",
        "description": (
            "Ask the Planner agent to decompose a task into a structured execution plan. "
            "Use this at the start of any complex, multi-step task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task or goal to plan",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant background context for the planner",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "call_researcher",
        "description": (
            "Ask the Research agent to gather information on a topic. "
            "Use this when you need factual information before implementing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The research question or topic to investigate",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context to guide the research",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "call_coder",
        "description": (
            "Ask the Coder agent to write code or an implementation. "
            "Provide clear specifications and any relevant context from research or planning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What to implement — be specific about language, requirements, and constraints",
                },
                "context": {
                    "type": "string",
                    "description": "Research findings, plan, or other context the Coder needs",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "call_reviewer",
        "description": (
            "Ask the Reviewer agent to review code or a plan for quality and correctness. "
            "Always call this after the Coder produces code."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The code or plan to review, plus the original requirements",
                },
                "context": {
                    "type": "string",
                    "description": "Original requirements or standards to review against",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "store_memory",
        "description": (
            "Store a key-value pair in shared memory so other agents can access it later. "
            "Use descriptive keys like 'research_findings', 'execution_plan', 'code_v1'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Memory key (use snake_case, descriptive)",
                },
                "value": {
                    "type": "string",
                    "description": "Content to store",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "read_memory",
        "description": "Retrieve a previously stored value from shared memory by key.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The memory key to retrieve",
                }
            },
            "required": ["key"],
        },
    },
]


class ManagerAgent:
    MAX_ITERATIONS = 20

    def __init__(self):
        self.name = "Manager"
        self.model = "claude-opus-4-7"
        self.client = anthropic.Anthropic()
        self.shared_memory = SharedMemory()
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()

    def orchestrate(self, user_query: str, verbose: bool = True) -> str:
        if verbose:
            print(f"\n[Manager] Starting orchestration for: {user_query[:80]}...")

        messages = [{"role": "user", "content": user_query}]
        iteration = 0

        while iteration < self.MAX_ITERATIONS:
            iteration += 1

            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": MANAGER_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
                tools=MANAGER_TOOLS,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final = next(
                    (b.text for b in response.content if b.type == "text"), ""
                )
                if verbose:
                    print(f"\n[Manager] Orchestration complete after {iteration} iteration(s).")
                return final

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if verbose:
                            print(f"\n[Manager] → Calling tool: {block.name}")
                        result = self._execute_tool(block.name, block.input, verbose)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return "Orchestration reached maximum iterations without completing."

    def _execute_tool(self, tool_name: str, tool_input: dict, verbose: bool) -> str:
        task = tool_input.get("task", "")
        context = tool_input.get("context", "")

        if tool_name == "call_planner":
            if verbose:
                print(f"  [Planner] Planning: {task[:60]}...")
            return self.planner.run(task, context)

        elif tool_name == "call_researcher":
            if verbose:
                print(f"  [Researcher] Researching: {task[:60]}...")
            return self.researcher.run_with_tools(task, context)

        elif tool_name == "call_coder":
            if verbose:
                print(f"  [Coder] Implementing: {task[:60]}...")
            return self.coder.run(task, context)

        elif tool_name == "call_reviewer":
            if verbose:
                print(f"  [Reviewer] Reviewing submission...")
            return self.reviewer.run(task, context)

        elif tool_name == "store_memory":
            key = tool_input.get("key", "")
            value = tool_input.get("value", "")
            self.shared_memory.write(key, value, author=self.name)
            if verbose:
                print(f"  [Memory] Stored key: '{key}'")
            return f"Successfully stored '{key}' in shared memory."

        elif tool_name == "read_memory":
            key = tool_input.get("key", "")
            value = self.shared_memory.read(key)
            if verbose:
                print(f"  [Memory] Read key: '{key}'")
            if value:
                return value
            return f"No entry found for key '{key}' in shared memory."

        return f"Unknown tool: {tool_name}"
