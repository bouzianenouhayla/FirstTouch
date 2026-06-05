from langsmith import traceable

from app.agents.base import BaseAgent

SYSTEM_PROMPT = (
    "You are a women's amateur football coach specialising in players who started late. "
    "Given a player's position, experience level, and available training time, "
    "create a practical weekly training plan with specific drills and focus areas. "
    "Be encouraging and realistic. Structure your answer with clear sections: "
    "weekly schedule, key drills, and one focus tip for the week. "
    "Keep it concise and actionable."
)


class TrainingAgent(BaseAgent):
    """Specialist agent that generates personalised training plans for amateur women players."""

    def __init__(self) -> None:
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            tools=[],
            config_name="training-agent",
        )

    def _get_tool_functions(self) -> dict[str, callable]:
        """Returns:
        Empty — training agent generates directly from Claude's knowledge.
        """
        return {}

    @traceable(name="training_agent.run", metadata={"agent": "TrainingAgent"})
    def run(self, question: str) -> tuple[str, list[str], float, float]:
        """Generate a personalised training plan.

        Args:
            question: Player profile and training request, e.g.
                      'I play midfield, train 3x per week, started 6 months ago'.

        Returns:
            Tuple of (training_plan, tools_called, retrieval_ms, total_ms).
        """
        return super().run(question)
