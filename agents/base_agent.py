import anthropic
from typing import Dict, List, Optional


class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        tools: Optional[List[Dict]] = None,
        model: str = "claude-opus-4-7",
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model
        self.memory: List[Dict] = []
        self.client = anthropic.Anthropic()

    @property
    def _cached_system(self) -> List[Dict]:
        return [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def run(self, task: str, context: str = "") -> str:
        user_message = f"{context}\n\nTask: {task}" if context else task
        messages = [{"role": "user", "content": user_message}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self._cached_system,
            messages=messages,
        )

        result = next(
            (b.text for b in response.content if b.type == "text"), ""
        )

        self.memory.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": result},
        ])
        return result

    def run_with_tools(self, task: str, context: str = "") -> str:
        user_message = f"{context}\n\nTask: {task}" if context else task
        messages = [{"role": "user", "content": user_message}]
        last_response = None

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._cached_system,
                messages=messages,
                tools=self.tools,
            )
            last_response = response
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._handle_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        if last_response:
            return next(
                (b.text for b in last_response.content if b.type == "text"), ""
            )
        return ""

    def _handle_tool(self, tool_name: str, tool_input: Dict) -> str:
        return f"Tool '{tool_name}' not implemented in {self.name}."
