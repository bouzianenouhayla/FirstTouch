FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv sync --no-group local --no-group eval --no-group dev

RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
