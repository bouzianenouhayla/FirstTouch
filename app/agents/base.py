import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class BaseAgent(ABC):
    """Shared tool-use loop for all agents in the multi-agent workflow."""

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        config_name: str,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Args:
        system_prompt: Static system instruction for this agent.
        tools: Tool definitions for this agent. Empty list for generation-only agents.
        config_name: Label used in tracing and results.
        model: Anthropic model ID.
        """
        self._system_prompt = system_prompt
        self.config_name = config_name
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._tools = tools[:]
        if self._tools:
            self._tools[-1] = {
                **self._tools[-1],
                "cache_control": {"type": "ephemeral"},
            }

    @abstractmethod
    def _get_tool_functions(self) -> dict[str, callable]:
        """Return a mapping of tool name to the callable that executes it.

        Returns:
            Dict mapping tool name strings to callables that accept the tool input dict.
        """

    def run(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[str, list[str], float, float]:
        """Execute the tool-use loop for this agent.

        Args:
            question: Natural language question or task for this agent.
            history: Prior conversation turns in Anthropic message format.
                     Prepended to messages so the agent has context from previous turns.

        Returns:
            Tuple of (answer, tools_called, retrieval_ms, total_ms).
        """
        messages: list[dict] = [*(history or []), {"role": "user", "content": question}]
        retrieval_ms = 0.0
        answer = "I could not answer that question."
        tools_called: list[str] = []
        t_start = time.perf_counter()
        tool_fns = self._get_tool_functions()

        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }
        if self._tools:
            create_kwargs["tools"] = self._tools

        while True:
            create_kwargs["messages"] = messages
            response = self._client.messages.create(**create_kwargs)

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        answer = block.text.strip()
                        break
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                t0 = time.perf_counter()
                for block in response.content:
                    if block.type == "tool_use":
                        tools_called.append(block.name)
                        result = tool_fns[block.name](block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                retrieval_ms += (time.perf_counter() - t0) * 1000
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        total_ms = (time.perf_counter() - t_start) * 1000
        return answer, tools_called, retrieval_ms, total_ms
