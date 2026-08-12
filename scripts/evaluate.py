"""
evaluate.py — Phase 7
=====================
Evaluates the RAG pipeline using RAGAS to measure:
- Faithfulness
- Answer Relevance
- Context Precision

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model gemini-3.1-flash-lite
"""

import os
import sys
import argparse
import math
from pathlib import Path

# ── Env must be loaded BEFORE any LLM/API imports so keys are available ──────
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# langchain-google-genai reads GOOGLE_API_KEY; map from our GEMINI_API_KEY if needed
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# ── Path setup so local modules are importable ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datasets import Dataset

from rag_pipeline import load_and_chunk_documents, build_index, rag_query
from test_rag import TEST_QUESTIONS

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ── Monkey-patch: strip `temperature` kwarg injected by RAGAS ────────────────
# RAGAS 0.1.x passes `temperature` directly to the underlying API call, but
# `langchain-google-genai<2.0` does not accept it as a keyword argument.
# Fix: intercept at the lowest-level `_generate` / `_agenerate` methods.

_orig_gen = ChatGoogleGenerativeAI._generate
def _patched_gen(self, messages, stop=None, run_manager=None, **kwargs):
    kwargs.pop("temperature", None)
    return _orig_gen(self, messages, stop=stop, run_manager=run_manager, **kwargs)
ChatGoogleGenerativeAI._generate = _patched_gen

_orig_agen = ChatGoogleGenerativeAI._agenerate
async def _patched_agen(self, messages, stop=None, run_manager=None, **kwargs):
    kwargs.pop("temperature", None)
    return await _orig_agen(self, messages, stop=stop, run_manager=run_manager, **kwargs)
ChatGoogleGenerativeAI._agenerate = _patched_agen

# ── Constants ─────────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "docs"
# Use a relative name so rag_pipeline resolves it from the project root, not an
# absolute Windows path (avoids cross-platform fragility).
CHROMA_DIR = "chroma_db"
DEFAULT_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "models/text-embedding-004"


def _make_ground_truth(q: dict) -> str:
    """
    Build a full-sentence ground truth from the test question record.

    The raw `key_fact` is often a short fragment (e.g. "30 calendar days").
    RAGAS context_precision and faithfulness perform significantly better when
    given a complete reference sentence.
    """
    question = q["question"].rstrip("?")
    key_fact = q["key_fact"]
    return f"{question}: {key_fact}."


def build_evaluation_dataset(
    collection,
    bm25,
    chunks: list[dict],
    questions: list[dict],
    model: str,
) -> Dataset:
    """
    Runs the RAG pipeline on answerable questions and builds a HuggingFace
    Dataset suitable for RAGAS evaluation.

    Skips:
    - Questions explicitly marked ``expected_answerable=False`` (unanswerable
      by design).
    - Questions where the RAG pipeline itself returns ``answerable=False``
      (no sufficient evidence found in the corpus), because an empty context
      list will produce meaningless RAGAS scores.
    - Questions that cause a pipeline exception (logged, then skipped so the
      rest of the evaluation is not aborted).
    """
    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    answerable_qs = [q for q in questions if q.get("expected_answerable", True)]
    print(f"\nBuilding evaluation dataset from {len(answerable_qs)} answerable questions...")

    for q in answerable_qs:
        print(f"  Processing Q: {q['question']}")

        try:
            result = rag_query(
                question=q["question"],
                collection=collection,
                bm25=bm25,
                chunks=chunks,
                model=model,
                k=4,
                search_method="hybrid",
                verbose=False,
            )
        except Exception as exc:
            print(f"    [SKIP] rag_query raised an exception: {exc}")
            continue

        # Skip rows where the pipeline found no evidence — empty contexts
        # cause RAGAS to silently return NaN for all metrics on that row.
        if not result.get("answerable", True):
            print(f"    [SKIP] RAG pipeline returned 'not answerable' (no evidence found)")
            continue

        context_texts: list[str] = [c["text"] for c in result["chunks"]]

        # Guard: if retrieved chunks somehow came back empty, skip the row.
        if not context_texts:
            print(f"    [SKIP] No chunks returned by retriever")
            continue

        data["question"].append(q["question"])
        data["answer"].append(result["answer"] or "")
        data["contexts"].append(context_texts)
        data["ground_truth"].append(_make_ground_truth(q))

    dataset = Dataset.from_dict(data)
    print(f"  Dataset built: {len(dataset)} rows")
    return dataset


def run_ragas_evaluation(dataset: Dataset) -> object:
    """
    Evaluates the dataset with RAGAS using Gemini as the judge LLM/embedder.
    """
    print("\nInitializing Gemini LLM and Embeddings for RAGAS...")
    llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]

    print(f"Running RAGAS evaluation on {len(dataset)} examples...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,   # surface per-job errors in progress bar, don't abort
    )
    return result


def _format_score(score) -> str:
    """Format a metric score safely, handling NaN."""
    try:
        if math.isnan(score):
            return "nan  [rate-limited or evaluation failed]"
    except (TypeError, ValueError):
        return str(score)
    return f"{score:.4f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with RAGAS")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model for answer generation (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    # 1. Load resources
    print("Loading documents and index...")
    chunks = load_and_chunk_documents(DOCS_DIR)
    collection, bm25, chunks = build_index(chunks, persist_dir=CHROMA_DIR, rebuild=False)

    # 2. Build evaluation dataset
    eval_dataset = build_evaluation_dataset(collection, bm25, chunks, TEST_QUESTIONS, args.model)

    if len(eval_dataset) == 0:
        sys.exit(
            "No valid questions found for evaluation. "
            "Check that rag_query returns answers and the pipeline can reach the API."
        )

    # 3. Run RAGAS evaluation
    eval_results = run_ragas_evaluation(eval_dataset)

    # 4. Display aggregate metrics (NaN-safe)
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION METRICS (Aggregate)")
    print("=" * 50)
    for metric_name, score in eval_results.items():
        print(f"  {metric_name.ljust(22)}: {_format_score(score)}")
    print("=" * 50)

    # 5. Save per-question detail to CSV
    df = eval_results.to_pandas()
    output_path = Path(__file__).parent.parent / "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed per-question results saved to: {output_path}")
