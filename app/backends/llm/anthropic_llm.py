import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

from .base import BaseLLM

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicLLM(BaseLLM):
    """LLM backend that calls the Anthropic API."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Args:
        model: Anthropic model ID to use for generation.
        """
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def build_prompt(self, system: str, context: str, question: str) -> dict:
        """Format inputs into the Anthropic messages structure.

        Args:
            system: System instruction string.
            context: Retrieved context text, or empty string if none.
            question: User question.

        Returns:
            Dict with 'system' and 'user' keys ready for the API call.
        """
        user_content = (
            f"Context:\n{context}\n\nQuestion: {question}" if context else question
        )
        return {"system": system, "user": user_content}

    @traceable(name="AnthropicLLM.generate")
    def generate(self, prompt: dict) -> str:
        """Call the Anthropic API and return the response text.

        Args:
            prompt: Dict produced by build_prompt() with 'system' and 'user' keys.

        Returns:
            Generated response string.
        """
        message = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": prompt["system"],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
        return message.content[0].text.strip()
