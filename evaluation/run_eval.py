import json
import logging
from datetime import datetime
from pathlib import Path

from app.backends.pipeline.base import BasePipeline
from app.rag_pipeline import RAGPipeline
from evaluation.eval_models import EvalResult, EvalRunSummary, EvalSample

EVAL_PATH = Path(__file__).with_name("eval_dataset.json")
RESULTS_DIR = Path(__file__).parent / "results"

logger = logging.getLogger(__name__)


def token_f1(prediction: str, reference: str) -> float:
    """Compute word-overlap F1 between prediction and reference.

    Args:
        prediction: Generated answer string.
        reference: Ground truth answer string.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_retrieval_recall(
    context_ids: list[str] | None, retrieved_contexts: list
) -> float:
    """Compute fraction of expected chunk IDs that were actually retrieved.

    Args:
        context_ids: Expected chunk IDs from the eval sample, or None if unspecified.
        retrieved_contexts: List of RetrievedContext objects from the pipeline.

    Returns:
        Recall score between 0.0 and 1.0. Returns 1.0 if context_ids is None.
    """
    if not context_ids:
        return 1.0
    retrieved_ids = {ctx.chunk_id for ctx in retrieved_contexts}
    hits = len(set(context_ids) & retrieved_ids)
    return hits / len(context_ids)


def load_dataset() -> list[EvalSample]:
    """Load eval samples from eval_dataset.json.

    Returns:
        List of validated EvalSample objects.
    """
    with open(EVAL_PATH, "r", encoding="utf-8") as fh:
        raw_samples = json.load(fh)
    return [EvalSample.model_validate(item) for item in raw_samples]


# Maps coordinator delegation tool names to their canonical expected_tool equivalents
# so multi-agent tool selection can be compared against the same eval labels.
_TOOL_ALIASES: dict[str, str] = {
    "ask_rules_agent": "search_laws",
    "ask_stats_agent": "search_stats",
    "ask_player_agent": "search_player",
}


def compute_tool_selection(
    expected_tool: str | None, tools_called: list[str]
) -> bool | None:
    """Check whether the agent called the expected tool.

    Args:
        expected_tool: Tool name from the eval sample, or None if unlabelled.
        tools_called: Tools the pipeline actually called. Coordinator delegation
                      tool names are resolved via _TOOL_ALIASES before matching.

    Returns:
        True/False if labelled, None otherwise.
    """
    if not expected_tool:
        return None
    resolved = {_TOOL_ALIASES.get(t, t) for t in tools_called}
    return expected_tool in resolved


def run(pipeline: BasePipeline | None = None) -> EvalRunSummary:
    """Run token F1, retrieval recall, and tool selection evaluation against all samples.

    Args:
        pipeline: BasePipeline to evaluate. Defaults to local Phi-3 + ChromaDB.

    Returns:
        EvalRunSummary with per-sample results and aggregate metrics.
    """
    if pipeline is None:
        pipeline = RAGPipeline(config_name="local-phi3-chroma")

    samples = load_dataset()
    results: list[EvalResult] = []

    for sample in samples:
        pipeline_result = pipeline.answer_question(sample.question)
        f1 = token_f1(pipeline_result.answer, sample.reference_answer)
        recall = compute_retrieval_recall(sample.context_ids, pipeline_result.contexts)
        tool_correct = compute_tool_selection(
            sample.expected_tool, pipeline_result.tools_called
        )

        results.append(
            EvalResult(
                sample_id=sample.id,
                config_name=pipeline_result.config_name,
                question=sample.question,
                answer=pipeline_result.answer,
                reference_answer=sample.reference_answer,
                answer_f1=f1,
                retrieval_recall=recall,
                total_time_ms=pipeline_result.total_time_ms,
                retrieval_time_ms=pipeline_result.retrieval_time_ms,
                llm_time_ms=pipeline_result.llm_time_ms,
                tool_selection_correct=tool_correct,
            )
        )

    summary = EvalRunSummary(config_name=pipeline.config_name, results=results)

    # Build the run entry to append
    tool_acc = summary.tool_selection_accuracy
    run_entry = {
        "config_name": summary.config_name,
        "timestamp": datetime.now().isoformat(),
        "avg_f1": round(summary.average_f1, 4),
        "avg_retrieval_recall": round(summary.average_retrieval_recall, 4),
        "avg_total_ms": round(summary.average_total_time_ms, 2),
        "avg_llm_ms": round(summary.average_llm_time_ms, 2),
        "tool_selection_accuracy": round(tool_acc, 4) if tool_acc is not None else None,
        "num_samples": len(results),
        "results": [r.model_dump() for r in results],
    }

    # Append to the shared results file — create it if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)
    all_runs_path = RESULTS_DIR / "all_runs.json"

    if all_runs_path.exists():
        all_runs = json.loads(all_runs_path.read_text(encoding="utf-8"))
    else:
        all_runs = []

    all_runs.append(run_entry)
    all_runs_path.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")

    logger.info(
        "Config=%s | Samples=%d | F1=%.3f | Recall=%.3f | ToolAcc=%s | "
        "Avg total=%.0fms | Avg LLM=%.0fms | Appended to %s",
        summary.config_name,
        len(results),
        summary.average_f1,
        summary.average_retrieval_recall,
        f"{tool_acc:.3f}" if tool_acc is not None else "n/a",
        summary.average_total_time_ms,
        summary.average_llm_time_ms,
        all_runs_path,
    )

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
