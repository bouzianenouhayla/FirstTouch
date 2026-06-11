import asyncio
from functools import partial
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .agent_pipeline import AgentPipeline
from .backends.llm.anthropic_llm import AnthropicLLM
from .backends.pipeline.base import BasePipeline
from .memory.store import MemoryStore
from .models import QueryRequest, QueryResponse, UserProfile
from .multi_agent_pipeline import MultiAgentPipeline
from .rag_pipeline import RAGPipeline

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
_memory = MemoryStore()

# Both pipelines are created once at startup — no overhead per request
_local_rag = RAGPipeline(config_name="local-phi3-chroma")

PIPELINES: dict[str, BasePipeline] = {
    "local": _local_rag,
    "anthropic": RAGPipeline(llm=AnthropicLLM(), config_name="anthropic-haiku-chroma"),
    "agent": AgentPipeline(rag=_local_rag),
    "multi-agent": MultiAgentPipeline(rag=_local_rag),
}


@router.get("/health")
def health() -> dict:
    """Return service health status and available backends.

    Returns:
        Dict with status and list of available backends.
    """
    return {"status": "ok", "backends": list(PIPELINES.keys())}


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Serve the chat UI.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Rendered index.html template.
    """
    return templates.TemplateResponse(request, "index.html")


@router.post("/ask", response_model=QueryResponse)
async def ask(query: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline non-blocking and return an answer.

    Runs the pipeline in a thread pool so the event loop is free to handle
    other requests while waiting for LLM responses.

    Args:
        query: Validated request with question, max_contexts, and backend.
               backend is 'local' (Phi-3), 'anthropic' (Haiku), or 'agent'.

    Returns:
        QueryResponse with answer, retrieved contexts, config name, and latency.
    """
    pipeline = PIPELINES[query.backend]
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            pipeline.answer_question,
            query.question,
            max_contexts=query.max_contexts,
            session_id=query.session_id,
        ),
    )
    return QueryResponse(
        answer=result.answer,
        contexts=result.contexts,
        config_name=result.config_name,
        total_time_ms=result.total_time_ms,
    )


@router.post("/profile")
async def save_profile(profile: UserProfile) -> dict:
    """Store a player profile as semantic memory facts.

    Converts structured profile fields into natural language facts and upserts
    them into the vector memory store so they are retrieved on every question.

    Args:
        profile: Validated player profile from the onboarding form.

    Returns:
        Dict with status and number of facts stored.
    """
    facts = [
        f"plays as a {profile.position}",
        f"{profile.level} level player",
        f"trains {profile.sessions_per_week} times per week",
        f"has been playing for {profile.playing_since}",
    ]
    if profile.goal.strip():
        facts.append(f"main goal is to {profile.goal.strip()}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: [_memory.store(f, profile.session_id) for f in facts],
    )
    return {"status": "ok", "facts_stored": len(facts)}
