import contextvars
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path

import anthropic
import structlog
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.resilience import CircuitBreaker, CircuitBreakerOpen

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_FALLBACK_ANSWER = "I'm temporarily unavailable. Please try again in a moment."
_TOOL_TIMEOUT_S = 30.0

logger = structlog.get_logger(__name__)


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
        self._breaker = CircuitBreaker(fail_max=5, reset_timeout=60.0)
        self._log = logger.bind(agent=config_name)

    @abstractmethod
    def _get_tool_functions(self) -> dict[str, Callable]:
        """Return a mapping of tool name to the callable that executes it.

        Returns:
            Dict mapping tool name strings to callables that accept the tool input dict.
        """

    def _call_api(
        self,
        *,
        messages: list[dict],
        system: list[dict],
    ) -> anthropic.types.Message | None:
        """Execute a single Anthropic API call guarded by the circuit breaker.

        Returns None on circuit-open or unrecoverable error so the caller can
        short-circuit immediately. Re-raises retryable Anthropic errors so
        tenacity can back off and retry them.

        Args:
            messages: Current message array for this turn.
            system: System prompt blocks (with optional user context appended).

        Returns:
            Anthropic Message, or None on fast-fail.
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": messages,
        }
        if self._tools:
            kwargs["tools"] = self._tools

        try:
            return self._breaker.call(self._client.messages.create, **kwargs)
        except CircuitBreakerOpen as exc:
            self._log.warning("circuit_open", detail=str(exc))
            return None
        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        ):
            raise
        except Exception as exc:
            self._log.error("api_error_unrecoverable", error=str(exc))
            return None

    def run(
        self,
        question: str,
        history: list[dict] | None = None,
        user_context: str | None = None,
    ) -> tuple[str, list[str], float, float]:
        """Execute the tool-use loop for this agent.

        Args:
            question: Natural language question or task for this agent.
            history: Prior conversation turns in Anthropic message format.
                     Prepended to messages so the agent has context from previous turns.
            user_context: Relevant facts about the user retrieved from semantic memory.
                          Appended as a second system block so the cached prefix is preserved.

        Returns:
            Tuple of (answer, tools_called, retrieval_ms, total_ms).
        """
        system: list[dict] = [
            {
                "type": "text",
                "text": self._system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if user_context:
            system.append(
                {
                    "type": "text",
                    "text": f"Known facts about this user:\n{user_context}",
                }
            )

        messages: list[dict] = [*(history or []), {"role": "user", "content": question}]
        retrieval_ms = 0.0
        answer = "I could not answer that question."
        tools_called: list[str] = []
        t_start = time.perf_counter()
        tool_fns = self._get_tool_functions()

        @retry(
            retry=retry_if_exception_type(
                (
                    anthropic.RateLimitError,
                    anthropic.APIConnectionError,
                    anthropic.InternalServerError,
                )
            ),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
            before_sleep=lambda s: self._log.warning(
                "api_retry", attempt=s.attempt_number
            ),
        )
        def _call() -> anthropic.types.Message | None:
            return self._call_api(messages=messages, system=system)

        # Copy the current context so tool threads inherit request_id and other
        # contextvars bound by the request middleware.
        ctx = contextvars.copy_context()

        while True:
            try:
                response = _call()
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as exc:
                self._log.error("api_retries_exhausted", error=str(exc))
                answer = _FALLBACK_ANSWER
                break

            if response is None:
                answer = _FALLBACK_ANSWER
                break

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if isinstance(block, anthropic.types.TextBlock):
                        answer = block.text.strip()
                        break
                break

            if response.stop_reason == "tool_use":
                tool_blocks = [b for b in response.content if b.type == "tool_use"]
                for block in tool_blocks:
                    tools_called.append(block.name)

                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=len(tool_blocks)) as executor:
                    futures = {
                        block.id: executor.submit(
                            ctx.run, tool_fns[block.name], block.input
                        )
                        for block in tool_blocks
                    }

                tool_results: list[dict] = []
                for block_id, future in futures.items():
                    try:
                        content = future.result(timeout=_TOOL_TIMEOUT_S)
                    except FutureTimeoutError:
                        self._log.warning("tool_timeout", tool_id=block_id)
                        content = "Tool timed out. The service may be slow — please try again."
                    except Exception as exc:
                        self._log.error("tool_error", tool_id=block_id, error=str(exc))
                        content = f"Tool execution failed: {exc}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": content,
                        }
                    )

                retrieval_ms += (time.perf_counter() - t0) * 1000
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        total_ms = (time.perf_counter() - t_start) * 1000
        return answer, tools_called, retrieval_ms, total_ms
