import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

from .backends.pipeline.base import BasePipeline
from .models import PipelineResult
from .rag_pipeline import RAGPipeline
from .tools.player_tool import search_player
from .tools.stats_tool import search_stats

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about women's football. "
    "Use search_laws for questions about rules and regulations. "
    "Use search_stats for questions about match results, scores, and team performance. "
    "Use search_player for questions about a specific player's appearances or career. "
    "Answer in 3 sentences or less. Do not ask follow-up questions."
)

ALL_TOOL_NAMES = ["search_laws", "search_stats", "search_player"]

TOOLS: list[dict] = [
    {
        "name": "search_laws",
        "description": (
            "Search the IFAB Laws of the Game for rules and regulations about football. "
            "Use this for questions about offside, handball, fouls, penalties, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query about football rules, e.g. 'offside rule'",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_stats",
        "description": (
            "Fetch live women's football match results and statistics from StatsBomb open data. "
            "Use this for questions about match scores, winners, tournaments, and team performance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query about match stats, e.g. 'FIFA Women World Cup 2023 results'",
                },
                "competition": {
                    "type": "string",
                    "description": "Optional competition name to narrow the search.",
                    "enum": [
                        "FIFA Women's World Cup",
                        "FA Women's Super League",
                        "NWSL",
                        "UEFA Women's Euro",
                    ],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_player",
        "description": (
            "Search a specific player's appearances in women's football from StatsBomb open data. "
            "Use this for questions about a named player — matches played, team, competition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {
                    "type": "string",
                    "description": "Player name to search for, e.g. 'Sam Kerr', 'Keira Walsh'",
                },
                "competition": {
                    "type": "string",
                    "description": "Optional competition name to narrow the search.",
                    "enum": [
                        "FIFA Women's World Cup",
                        "FA Women's Super League",
                        "NWSL",
                        "UEFA Women's Euro",
                    ],
                },
            },
            "required": ["player_name"],
        },
    },
]


class AgentPipeline(BasePipeline):
    """Agentic pipeline that uses Anthropic tool use to pick between law search and live stats."""

    def __init__(
        self,
        rag: RAGPipeline | None = None,
        model: str = DEFAULT_MODEL,
        tools: list[str] | None = None,
    ) -> None:
        """Args:
        rag: RAGPipeline instance used for law retrieval. Created with defaults if not provided.
        model: Anthropic model ID to use.
        tools: List of tool names to enable. Defaults to all tools.
               Valid values: 'search_laws', 'search_stats'.
        """
        self._rag = rag or RAGPipeline()
        self.model = model
        enabled = tools if tools is not None else ALL_TOOL_NAMES
        self._tools = [t for t in TOOLS if t["name"] in enabled]
        if self._tools:
            self._tools[-1] = {
                **self._tools[-1],
                "cache_control": {"type": "ephemeral"},
            }
        self.config_name = f"agent-{'_'.join(enabled)}-{model}"
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @traceable(name="agent_answer", metadata={"pipeline": "AgentPipeline"})
    def answer_question(self, question: str, **_kwargs) -> PipelineResult:
        """Run the agentic tool-use loop and return a structured result.

        Args:
            question: Natural language question from the user.

        Returns:
            PipelineResult with answer and timing. contexts is empty (tool use replaces retrieval).
        """
        t_start = time.perf_counter()
        messages: list[dict] = [{"role": "user", "content": question}]
        retrieval_ms = 0.0
        answer = "I could not answer that question."
        tools_called: list[str] = []

        def _search_laws(args: dict) -> str:
            contexts = self._rag.retrieve(args["query"])
            if not contexts:
                return "No relevant rules found."
            return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in contexts)

        tool_functions = {
            "search_laws": _search_laws,
            "search_stats": lambda args: search_stats(
                args["query"], args.get("competition")
            ),
            "search_player": lambda args: search_player(
                args["player_name"], args.get("competition")
            ),
        }

        while True:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=self._tools,
                messages=messages,
            )

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
                        result = tool_functions[block.name](block.input)
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
        llm_ms = total_ms - retrieval_ms

        return PipelineResult(
            answer=answer,
            contexts=[],
            config_name=self.config_name,
            total_time_ms=round(total_ms, 2),
            retrieval_time_ms=round(retrieval_ms, 2),
            llm_time_ms=round(llm_ms, 2),
            retrieval_used=True,
            chunks_retrieved=0,
            tools_called=tools_called,
        )
