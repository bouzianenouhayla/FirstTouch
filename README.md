# FirstTouch

A football coach in your pocket — built for women who play, at any level, at any age.

Ask about the rules, follow your favourite players, explore match stats. FirstTouch combines a RAG pipeline over the IFAB Laws of the Game with live StatsBomb women's football data, all powered by an agentic AI that picks the right tool for your question.

## What it does

- **Rules Q&A** — ask anything about the Laws of the Game in plain language
- **Live stats** — match results, scores, and standings from real women's competitions (WWC, WSL, NWSL, UEFA Women's Euro)
- **Player search** — find out where a player appeared, which matches she played, which team she represented
- **Agentic routing** — the AI decides whether to search the laws, fetch live data, or look up a player

## Stack

- **FastAPI** backend with a lightweight chat UI
- **ChromaDB** local vector store
- **SentenceTransformer** embeddings (`all-MiniLM-L6-v2`)
- **Phi-3 mini** (GGUF via `llama-cpp-python`) — local LLM
- **Claude Haiku** (Anthropic API) — API LLM
- **statsbombpy** — live women's football data
- **LangSmith** tracing on all pipeline steps

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
ANTHROPIC_API_KEY=...
LANGSMITH_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=firsttouch
```

Place your GGUF model under `models/` (e.g. `phi-3-mini.gguf`).

Ingest the Laws of the Game into ChromaDB:

```bash
python ingestion/ingest.py
```

Run the app:

```bash
python main.py
```

## Evaluation

Run a single pipeline config:

```bash
python -m evaluation.compare --config agent-all
```

Available configs: `rag`, `llm-only`, `rag-strict`, `anthropic-rag`, `anthropic-llm-only`, `agent-laws`, `agent-stats`, `agent-player`, `agent-all`

Results are appended to `evaluation/results/all_runs.json`.

Metrics: token F1, retrieval recall, tool selection accuracy.

## Tests

```bash
pytest
```

## Code Quality

Pre-commit hooks run ruff on every commit:

```bash
pip install pre-commit && pre-commit install
```
