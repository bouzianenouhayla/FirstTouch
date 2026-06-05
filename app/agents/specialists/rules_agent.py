from langsmith import traceable

from app.agents.base import BaseAgent
from app.rag_pipeline import RAGPipeline

SYSTEM_PROMPT = (
    "You are a women's football rules expert. "
    "Use search_laws to look up IFAB Laws of the Game and answer questions about rules, "
    "fouls, offsides, penalties, and regulations. "
    "Answer in 3 sentences or less. Cite the relevant law when possible."
)

TOOLS: list[dict] = [
    {
        "name": "search_laws",
        "description": "Search the IFAB Laws of the Game for rules and regulations.",
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
    }
]


class RulesAgent(BaseAgent):
    """Specialist agent for IFAB Laws of the Game queries."""

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        """Args:
        rag: RAGPipeline instance for law retrieval. Created with defaults if not provided.
        """
        super().__init__(
            system_prompt=SYSTEM_PROMPT, tools=TOOLS, config_name="rules-agent"
        )
        self._rag = rag or RAGPipeline()

    def _get_tool_functions(self) -> dict[str, callable]:
        """Returns:
        Tool function mapping for search_laws.
        """

        def _search_laws(args: dict) -> str:
            contexts = self._rag.retrieve(args["query"])
            if not contexts:
                return "No relevant rules found."
            return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in contexts)

        return {"search_laws": _search_laws}

    @traceable(name="rules_agent.run", metadata={"agent": "RulesAgent"})
    def run(self, question: str) -> tuple[str, list[str], float, float]:
        """Run the rules agent tool-use loop.

        Args:
            question: Football rules question.

        Returns:
            Tuple of (answer, tools_called, retrieval_ms, total_ms).
        """
        return super().run(question)
