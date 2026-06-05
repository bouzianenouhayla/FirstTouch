from langsmith import traceable

from app.agents.base import BaseAgent
from app.tools.player_tool import search_player

SYSTEM_PROMPT = (
    "You are a women's football player expert. "
    "Use search_player to find a player's match appearances, teams, and competitions. "
    "Answer in 3 sentences or less. Include competition and team details."
)

TOOLS: list[dict] = [
    {
        "name": "search_player",
        "description": (
            "Search a specific player's appearances in women's football from StatsBomb open data."
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
    }
]


class PlayerAgent(BaseAgent):
    """Specialist agent for women's football player queries."""

    def __init__(self) -> None:
        super().__init__(
            system_prompt=SYSTEM_PROMPT, tools=TOOLS, config_name="player-agent"
        )

    def _get_tool_functions(self) -> dict[str, callable]:
        """Returns:
        Tool function mapping for search_player.
        """
        return {
            "search_player": lambda args: search_player(
                args["player_name"], args.get("competition")
            )
        }

    @traceable(name="player_agent.run", metadata={"agent": "PlayerAgent"})
    def run(self, question: str) -> tuple[str, list[str], float, float]:
        """Run the player agent tool-use loop.

        Args:
            question: Question about a specific player.

        Returns:
            Tuple of (answer, tools_called, retrieval_ms, total_ms).
        """
        return super().run(question)
