import time

from langsmith import traceable

from ingestion.base import BaseEmbedder, BaseVectorStore
from ingestion.embedder import LocalEmbedder
from ingestion.vectorstores.chroma import ChromaStore

from .backends.llm.base import BaseLLM
from .backends.pipeline.base import BasePipeline
from .backends.llm.local_llm import LocalLLM
from .models import PipelineResult, RetrievedContext

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about women's football using the provided "
    "context. Cite key laws or data points from the context. "
    "Answer in 3 sentences or less. Do not ask follow-up questions."
)

SYSTEM_PROMPT_NO_CONTEXT = (
    "You are a helpful assistant that answers questions about women's football. "
    "Answer in 3 sentences or less. Do not ask follow-up questions."
)


class RAGPipeline(BasePipeline):
    """Orchestrates retrieval and generation using pluggable backends."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: BaseVectorStore | None = None,
        llm: BaseLLM | None = None,
        config_name: str = "default",
        use_retrieval: bool = True,
        min_score: float = 0.0,
    ) -> None:
        """Args:
        embedder: Converts query text to vectors. Defaults to LocalEmbedder.
        store: Vector database to retrieve chunks from. Defaults to ChromaStore.
        llm: Language model for generation. Defaults to LocalLLM.
        config_name: Label for this configuration, used in eval results.
        use_retrieval: If False, skip retrieval and answer from the LLM alone.
        min_score: Minimum cosine similarity for a chunk to be included (0–1).
        """
        self.embedder = embedder or LocalEmbedder()
        self.store = store or ChromaStore()
        self.llm = llm or LocalLLM()
        self.config_name = config_name
        self.use_retrieval = use_retrieval
        self.min_score = min_score

    @traceable(name="retrieve")
    def retrieve(self, question: str, k: int = 5) -> list[RetrievedContext]:
        """Embed the question and fetch the k most similar chunks above min_score.

        Args:
            question: Natural language question to retrieve context for.
            k: Maximum number of chunks to retrieve.

        Returns:
            List of RetrievedContext objects that passed the score threshold.
        """
        embedding = self.embedder.embed_text(question)
        raw = self.store.query(embedding=embedding, k=k)
        all_contexts = [RetrievedContext(**r) for r in raw]
        return [c for c in all_contexts if c.score >= self.min_score]

    @traceable(name="answer_question", metadata={"pipeline": "RAGPipeline"})
    def answer_question(
        self, question: str, max_contexts: int = 5, **_kwargs
    ) -> PipelineResult:
        """Run the full pipeline and return a timed, structured result.

        Args:
            question: Natural language question to answer.
            max_contexts: Maximum number of retrieved chunks to include in the prompt.

        Returns:
            PipelineResult containing the answer, contexts, config name, and timing.
        """
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        contexts = self.retrieve(question, k=max_contexts) if self.use_retrieval else []
        retrieval_ms = (time.perf_counter() - t0) * 1000

        if contexts:
            context_text = "\n\n".join(
                f"[Chunk {c.chunk_id}] {c.text}" for c in contexts
            )
            prompt = self.llm.build_prompt(
                system=SYSTEM_PROMPT,
                context=context_text,
                question=question,
            )
        else:
            prompt = self.llm.build_prompt(
                system=SYSTEM_PROMPT_NO_CONTEXT,
                context="",
                question=question,
            )

        t0 = time.perf_counter()
        answer = self.llm.generate(prompt)
        llm_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000

        return PipelineResult(
            answer=answer,
            contexts=contexts,
            config_name=self.config_name,
            total_time_ms=round(total_ms, 2),
            retrieval_time_ms=round(retrieval_ms, 2),
            llm_time_ms=round(llm_ms, 2),
            retrieval_used=self.use_retrieval,
            chunks_retrieved=len(contexts),
        )
