from langsmith import traceable

from app.agents.base import BaseAgent
from app.tools.stats_tool import search_stats

SYSTEM_PROMPT = (
    "You are a women's football statistics analyst. "
    "Use search_stats to fetch live match results and standings from real competitions. "
    "Answer in 3 sentences or less. Include scores and team names when relevant."
)

TOOLS: list[dict] = [
    {
        "name": "search_stats",
        "description": (
            "Fetch live women's football match results from StatsBomb open data. "
            "Use for questions about scores, winners, and team performance."
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
    }
]


class StatsAgent(BaseAgent):
    """Specialist agent for women's football match statistics."""

    def __init__(self) -> None:
        super().__init__(
            system_prompt=SYSTEM_PROMPT, tools=TOOLS, config_name="stats-agent"
        )

    def _get_tool_functions(self) -> dict[str, callable]:
        """Returns:
        Tool function mapping for search_stats.
        """
        return {
            "search_stats": lambda args: search_stats(
                args["query"], args.get("competition")
            )
        }

    @traceable(name="stats_agent.run", metadata={"agent": "StatsAgent"})
    def run(self, question: str) -> tuple[str, list[str], float, float]:
        """Run the stats agent tool-use loop.

        Args:
            question: Match stats or results question.

        Returns:
            Tuple of (answer, tools_called, retrieval_ms, total_ms).
        """
        return super().run(question)
