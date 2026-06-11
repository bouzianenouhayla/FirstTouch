import json
import os
from pathlib import Path

import anthropic
import structlog
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
logger = structlog.get_logger(__name__)

_SYSTEM = (
    "Extract concrete facts about the user from this conversation exchange. "
    "Return a JSON array of short fact strings. "
    "Only extract personal facts: position, experience level, training frequency, physical attributes, goals. "
    "Do not extract questions or general football knowledge. "
    "If nothing can be extracted, return []. "
    'Example: ["plays as a midfielder", "trains 3 times per week", "started playing 6 months ago"]'
)


class MemoryExtractor:
    """Uses Claude to extract user facts from conversation exchanges."""

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def extract(self, question: str, answer: str) -> list[str]:
        """Extract user facts from a single conversation turn.

        Args:
            question: The user's question.
            answer: The assistant's response.

        Returns:
            List of extracted fact strings. Empty list if none found or on error.
        """
        try:
            response = self._client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=256,
                system=_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"User: {question}\nAssistant: {answer}",
                    }
                ],
            )
            text = response.content[0].text.strip()
            text = (
                text.removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            return json.loads(text)
        except Exception as exc:
            logger.debug("memory_extraction_failed", error=str(exc))
            return []
