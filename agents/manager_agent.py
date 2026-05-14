import json
import re
import time
from groq import Groq, BadRequestError, RateLimitError

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
        "type": "function",
        "function": {
            "name": "call_planner",
            "description": (
                "Ask the Planner agent to decompose a task into a structured execution plan. "
                "Use this at the start of any complex, multi-step task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task or goal to plan"},
                    "context": {"type": "string", "description": "Relevant background context"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_researcher",
            "description": (
                "Ask the Research agent to gather information on a topic. "
                "Use this when you need factual information before implementing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The research question or topic"},
                    "context": {"type": "string", "description": "Additional context to guide research"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_coder",
            "description": (
                "Ask the Coder agent to write code or an implementation. "
                "Provide clear specifications and any relevant context from research or planning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to implement — be specific about language and requirements"},
                    "context": {"type": "string", "description": "Research findings, plan, or other context"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_reviewer",
            "description": (
                "Ask the Reviewer agent to review code or a plan for quality and correctness. "
                "Always call this after the Coder produces code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The code or plan to review, plus original requirements"},
                    "context": {"type": "string", "description": "Original requirements to review against"},
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": (
                "Store a key-value pair in shared memory so other agents can access it later. "
                "Use descriptive keys like 'research_findings', 'execution_plan', 'code_v1'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key (use snake_case, descriptive)"},
                    "value": {"type": "string", "description": "Content to store"},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Retrieve a previously stored value from shared memory by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The memory key to retrieve"},
                },
                "required": ["key"],
            },
        },
    },
]


class ManagerAgent:
    MAX_ITERATIONS = 10

    def __init__(self):
        self.name = "Manager"
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq()
        self.shared_memory = SharedMemory()
        self.planner = PlannerAgent()
        self.researcher = ResearchAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()

    @staticmethod
    def _trim_messages(messages: list, window: int = 12) -> list:
        """Keep system + first user message, then the last `window` messages."""
        if len(messages) <= window + 2:
            return messages
        return messages[:2] + messages[-window:]

    @staticmethod
    def _parse_llama_tool_calls(text: str) -> list:
        """Parse all Llama-style function call variants from text."""
        calls = []
        # Try each known format in order of specificity
        for pattern in [
            r'<function=(\w+)\((\{.+?\})\)\s*>',        # <function=name(json)>
            r'<function=(\w+),(\{.+?\})\s*</function>',  # <function=name,json</function>
            r'<function=(\w+),(\{.+?\})\s*>',            # <function=name,json>
            r'<function=(\w+),(\{.+?\})',                 # <function=name,json  (no closing)
        ]:
            for m in re.finditer(pattern, text, re.DOTALL):
                try:
                    calls.append((m.group(1), json.loads(m.group(2))))
                except json.JSONDecodeError:
                    pass
            if calls:
                break
        return calls

    def _recover_from_bad_request(self, error: BadRequestError, messages: list, verbose: bool) -> list | None:
        """If Groq returns tool_use_failed, parse failed_generation and inject results."""
        body = getattr(error, "body", {}) or {}
        failed_gen = body.get("error", {}).get("failed_generation", "")
        if not failed_gen:
            return None

        calls = self._parse_llama_tool_calls(failed_gen)
        if not calls:
            return None

        tool_results = []
        for name, args in calls:
            if verbose:
                print(f"\n[Manager] Recovering malformed tool call: {name}")
            result = self._execute_tool(name, args, verbose)
            if len(result) > 1500:
                result = result[:1500] + "\n\n[...truncated...]"
            tool_results.append(f"Tool '{name}' result:\n{result}")

        messages.append({
            "role": "user",
            "content": (
                "Tool execution results:\n"
                + "\n\n".join(tool_results)
                + "\n\nPlease continue the orchestration based on these results."
            ),
        })
        return messages

    def orchestrate(self, user_query: str, verbose: bool = True) -> str:
        if verbose:
            print(f"\n[Manager] Starting orchestration for: {user_query[:80]}...")

        messages = [
            {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        iteration = 0
        last_text = ""

        while iteration < self.MAX_ITERATIONS:
            iteration += 1

            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self._trim_messages(messages),
                        tools=MANAGER_TOOLS,
                        tool_choice="auto",
                    )
                    break
                except RateLimitError as e:
                    wait = 2 ** attempt
                    if verbose:
                        print(f"\n[Manager] Rate limited — retrying in {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:
                        return last_text or str(e)
                except BadRequestError as e:
                    recovered = self._recover_from_bad_request(e, messages, verbose)
                    if recovered is not None:
                        messages = recovered
                        response = None
                        break
                    if verbose:
                        print(f"\n[Manager] Unrecoverable error: {e}")
                    return last_text or str(e)
            else:
                continue

            if response is None:
                continue

            choice = response.choices[0]
            message = choice.message

            if message.content:
                last_text = message.content

            assistant_msg: dict = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            if choice.finish_reason == "stop":
                if verbose:
                    print(f"\n[Manager] Orchestration complete after {iteration} iteration(s).")
                return message.content or last_text

            if choice.finish_reason == "tool_calls" and message.tool_calls:
                for tc in message.tool_calls:
                    if verbose:
                        print(f"\n[Manager] → Calling tool: {tc.function.name}")
                    tool_input = json.loads(tc.function.arguments)
                    result = self._execute_tool(tc.function.name, tool_input, verbose)
                    if len(result) > 1500:
                        result = result[:1500] + "\n\n[...truncated...]"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                break

        return last_text or "Orchestration reached maximum iterations without completing."

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
