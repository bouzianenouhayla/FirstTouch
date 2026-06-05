from langsmith import traceable

from app.agents.base import BaseAgent
from app.agents.specialists.player_agent import PlayerAgent
from app.agents.specialists.rules_agent import RulesAgent
from app.agents.specialists.stats_agent import StatsAgent
from app.agents.specialists.training_agent import TrainingAgent
from app.rag_pipeline import RAGPipeline

SYSTEM_PROMPT = (
    "You are a coordinator for a women's football assistant. "
    "Route every question to the right specialist using the delegation tools: "
    "- ask_rules_agent: rules, laws, fouls, offsides, penalties "
    "- ask_stats_agent: match results, scores, standings, team performance "
    "- ask_player_agent: questions about a specific named player "
    "- ask_training_agent: training plans, drills, improving skills "
    "Always delegate — never answer directly. Use the specialist's response as your final answer."
)

TOOLS: list[dict] = [
    {
        "name": "ask_rules_agent",
        "description": "Delegate to the rules specialist for questions about IFAB Laws of the Game.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The rules question to answer",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_stats_agent",
        "description": "Delegate to the stats specialist for match results, scores, and standings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The stats question to answer",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_player_agent",
        "description": "Delegate to the player specialist for questions about a specific named player.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The player question to answer",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "ask_training_agent",
        "description": "Delegate to the training coach for personalised training plans and drills.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Player profile and training request",
                }
            },
            "required": ["query"],
        },
    },
]


class CoordinatorAgent(BaseAgent):
    """Mediator agent that routes questions to the appropriate specialist."""

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        """Args:
        rag: RAGPipeline passed through to the RulesAgent for law retrieval.
        """
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            tools=TOOLS,
            config_name="coordinator",
        )
        self._rules = RulesAgent(rag=rag)
        self._stats = StatsAgent()
        self._player = PlayerAgent()
        self._training = TrainingAgent()

    def _get_tool_functions(self) -> dict[str, callable]:
        """Returns:
        Tool function mapping — each tool invokes the corresponding specialist agent.
        """
        return {
            "ask_rules_agent": lambda args: self._rules.run(args["query"])[0],
            "ask_stats_agent": lambda args: self._stats.run(args["query"])[0],
            "ask_player_agent": lambda args: self._player.run(args["query"])[0],
            "ask_training_agent": lambda args: self._training.run(args["query"])[0],
        }

    @traceable(name="coordinator.run", metadata={"agent": "CoordinatorAgent"})
    def run(
        self, question: str, history: list[dict] | None = None
    ) -> tuple[str, list[str], float, float]:
        """Route the question to the right specialist and return the answer.

        Args:
            question: User's natural language question.
            history: Prior conversation turns to give the coordinator context.

        Returns:
            Tuple of (answer, specialists_called, retrieval_ms, total_ms).
        """
        return super().run(question, history=history)
