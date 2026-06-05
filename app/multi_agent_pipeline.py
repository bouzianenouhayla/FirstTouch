from pathlib import Path
from uuid import uuid4

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langsmith import traceable

from app.agents.coordinator import CoordinatorAgent
from app.backends.pipeline.base import BasePipeline
from app.models import PipelineResult
from app.rag_pipeline import RAGPipeline

_HISTORY_DB = f"sqlite:///{Path(__file__).resolve().parents[1] / 'chat_history.db'}"
_MAX_HISTORY_TURNS = 5


def _load_history(session_id: str) -> list[dict]:
    """Load the last N turns from the session history as Anthropic-format messages.

    Args:
        session_id: Unique session identifier.

    Returns:
        List of role/content dicts ready to prepend to the messages array.
    """
    store = SQLChatMessageHistory(session_id=session_id, connection_string=_HISTORY_DB)
    messages = []
    for msg in store.messages[-_MAX_HISTORY_TURNS * 2 :]:
        if msg.type == "human":
            messages.append({"role": "user", "content": msg.content})
        elif msg.type == "ai":
            messages.append({"role": "assistant", "content": msg.content})
    return messages


def _save_turn(session_id: str, question: str, answer: str) -> None:
    """Persist a completed turn to the session history.

    Args:
        session_id: Unique session identifier.
        question: The user's question.
        answer: The agent's answer.
    """
    store = SQLChatMessageHistory(session_id=session_id, connection_string=_HISTORY_DB)
    store.add_user_message(question)
    store.add_ai_message(answer)


class MultiAgentPipeline(BasePipeline):
    """Entry point for the multi-agent workflow. Implements BasePipeline for eval compatibility."""

    def __init__(self, rag: RAGPipeline | None = None) -> None:
        """Args:
        rag: RAGPipeline passed through to the RulesAgent for law retrieval.
        """
        self._coordinator = CoordinatorAgent(rag=rag)
        self.config_name = "multi-agent"

    @traceable(name="multi_agent_answer", metadata={"pipeline": "MultiAgentPipeline"})
    def answer_question(self, question: str, **kwargs) -> PipelineResult:
        """Route the question through the coordinator and return a structured result.

        Args:
            question: Natural language question from the user.
            **kwargs: Accepts session_id (str) for conversation history. A new
                      session is created if not provided.

        Returns:
            PipelineResult with answer and timing. contexts is empty (agents handle retrieval).
        """
        session_id = kwargs.get("session_id") or str(uuid4())
        history = _load_history(session_id)

        answer, specialists_called, retrieval_ms, total_ms = self._coordinator.run(
            question, history=history
        )

        _save_turn(session_id, question, answer)

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
            tools_called=specialists_called,
        )
