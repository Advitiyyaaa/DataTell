"""
test_rag.py — Phase 3
======================
Tests the RAG pipeline against 10 policy questions across all 4 documents.
Also runs a head-to-head comparison: hybrid vs pure-vector vs pure-BM25
to demonstrate why hybrid is better.

Usage (from project root):
    python scripts/test_rag.py
    python scripts/test_rag.py --compare   # also run vector/BM25 head-to-head
    python scripts/test_rag.py --rebuild   # force rebuild of ChromaDB index
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import (
    load_and_chunk_documents,
    build_index,
    rag_query,
    hybrid_search,
    vector_search,
    bm25_search,
)

DOCS_DIR = Path(__file__).parent.parent / "docs"
CHROMA_DIR = "chroma_db"
MODEL = "gemini-flash-latest"

# ---------------------------------------------------------------------------
# 10 Test questions — mapped to expected source documents
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the return window for defective products?",
        "expected_sources": ["return_policy.md"],
        "key_fact": "30 calendar days",
    },
    {
        "id": 2,
        "question": "How long does standard delivery take to remote areas?",
        "expected_sources": ["shipping_policy.md"],
        "key_fact": "8 to 15 business days",
    },
    {
        "id": 3,
        "question": "What payment methods does Olist accept?",
        "expected_sources": ["customer_faq.md"],
        "key_fact": "credit card, boleto, debit card, PIX, vouchers",
    },
    {
        "id": 4,
        "question": "What review score must sellers maintain to stay active on the platform?",
        "expected_sources": ["seller_guidelines.md"],
        "key_fact": "3.5 minimum",
    },
    {
        "id": 5,
        "question": "Can I cancel an order after it has already been shipped?",
        "expected_sources": ["customer_faq.md"],
        "key_fact": "No — once dispatched, cancellation is not possible",
    },
    {
        "id": 6,
        "question": "What are the conditions for returning a purchased item?",
        "expected_sources": ["return_policy.md"],
        "key_fact": "unused, original packaging, within 7 days",
    },
    {
        "id": 7,
        "question": "Within how many business days must sellers dispatch orders?",
        "expected_sources": ["shipping_policy.md", "seller_guidelines.md"],
        "key_fact": "3 business days",
    },
    {
        "id": 8,
        "question": "What happens to sellers with consistently low review scores?",
        "expected_sources": ["seller_guidelines.md"],
        "key_fact": "suspension below 3.0 or below 3.5 for 60 days",
    },
    {
        "id": 9,
        "question": "How is a boleto refund processed and how long does it take?",
        "expected_sources": ["return_policy.md"],
        "key_fact": "bank transfer within 5 business days",
    },
    {
        "id": 10,
        "question": "What legal warranty period applies to electronics under Brazilian law?",
        "expected_sources": ["return_policy.md", "customer_faq.md"],
        "key_fact": "90 days",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_rag_tests(
    collection,
    bm25,
    chunks: list[dict],
    questions: list[dict],
    model: str = MODEL,
    retrieval_only: bool = False,
) -> list[dict]:
    """Run all test questions and return result records.
    
    retrieval_only=True: only evaluate chunk retrieval, skip LLM generation.
    Useful when API quota is exhausted or models are unavailable.
    """
    results = []
    total = len(questions)
    mode = "RETRIEVAL ONLY" if retrieval_only else f"Model: {model}"

    print(f"\n{'='*70}")
    print(f"  DataTell Phase 3 -- RAG Pipeline Test Suite")
    print(f"  {mode} | Questions: {total} | Method: hybrid")
    print(f"{'='*70}\n")

    for q in questions:
        print(f"[{q['id']:02d}/{total}] {q['question']}")
        print(f"         Expected key fact: {q['key_fact']}")

        # Retrieval only: evaluate chunk sources, skip generation
        if retrieval_only:
            retrieved = hybrid_search(q["question"], collection, bm25, chunks, k=4)
            sources = sorted(set(c["metadata"].get("source", "?") for c in retrieved))
            expected_sources = set(q["expected_sources"])
            source_hit = bool(set(sources) & expected_sources)
            status = "[PASS]" if source_hit else "[MISS]"
            print(f"         {status} | Retrieved sources: {sources}")
            print(f"         Expected sources : {q['expected_sources']}")
            print(f"         Top chunk: {retrieved[0]['text'][:120].replace(chr(10), ' ')}...\n")
            results.append({
                "id": q["id"], "question": q["question"],
                "status": status, "source_hit": source_hit, "fact_hit": False,
                "sources": sources, "retrieval_ms": 0, "generation_ms": 0,
                "answer_snippet": "[retrieval only mode]",
            })
            continue

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
            print(f"         [ERROR] {exc}")
            results.append({
                "id": q["id"],
                "question": q["question"],
                "status": "[ERROR]",
                "source_hit": False,
                "fact_hit": False,
                "sources": [],
                "retrieval_ms": 0,
                "generation_ms": 0,
                "answer_snippet": str(exc)[:200],
            })
            print()
            time.sleep(15)
            continue

        # Check if expected sources were retrieved
        retrieved_sources = set(result["sources"])
        expected_sources = set(q["expected_sources"])
        source_hit = bool(retrieved_sources & expected_sources)

        # Check if key fact appears in answer (simple heuristic)
        key_terms = q["key_fact"].lower().split()
        answer_lower = result["answer"].lower()
        fact_hit = sum(1 for t in key_terms if t in answer_lower) >= len(key_terms) // 2

        status = "[PASS]" if (source_hit and fact_hit) else "[PARTIAL]" if source_hit else "[MISS]"

        print(f"         {status} | Sources: {result['sources']}")
        print(f"         Timing: retrieval={result['retrieval_ms']}ms | generation={result['generation_ms']}ms")
        print(f"\n         Answer:\n         {result['answer'][:400]}{'...' if len(result['answer']) > 400 else ''}\n")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "status": status,
            "source_hit": source_hit,
            "fact_hit": fact_hit,
            "sources": result["sources"],
            "retrieval_ms": result["retrieval_ms"],
            "generation_ms": result["generation_ms"],
            "answer_snippet": result["answer"][:200],
        })

        # Longer pause between questions — free tier is 20 req/day per model
        time.sleep(15)

    return results


def run_comparison(query: str, collection, bm25, chunks):
    """
    Compare hybrid vs vector vs BM25 retrieval for a single query.
    Demonstrates why hybrid search is the differentiator.
    """
    print(f"\n{'='*70}")
    print(f"  RETRIEVAL COMPARISON for: '{query}'")
    print(f"{'='*70}\n")

    for method_name, method_fn in [
        ("HYBRID (BM25 + Vector + RRF)", lambda: hybrid_search(query, collection, bm25, chunks, k=3)),
        ("VECTOR only", lambda: vector_search(query, collection, k=3)),
        ("BM25 only", lambda: bm25_search(query, bm25, chunks, k=3)),
    ]:
        results = method_fn()
        print(f"--- {method_name} ---")
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            score = r["score"]
            preview = r["text"][:100].replace("\n", " ")
            print(f"  [{i}] {src} | score={score:.3f}")
            print(f"       {preview}...")
        print()


def print_summary(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "[PASS]")
    partial = sum(1 for r in results if r["status"] == "[PARTIAL]")
    missed = sum(1 for r in results if r["status"] == "[MISS]")
    errored = sum(1 for r in results if r["status"] == "[ERROR]")
    ran = total - errored
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total     : {total}")
    print(f"  Pass      : {passed} ({passed/max(ran,1)*100:.0f}% of {ran} completed)")
    print(f"  Partial   : {partial}")
    print(f"  Miss      : {missed}")
    if errored:
        print(f"  Errors    : {errored} (quota/network)")
    avg_retrieval = sum(r["retrieval_ms"] for r in results if r["retrieval_ms"]) / max(ran, 1)
    avg_generation = sum(r["generation_ms"] for r in results if r["generation_ms"]) / max(ran, 1)
    print(f"  Avg retrieval  : {avg_retrieval:.0f}ms")
    print(f"  Avg generation : {avg_generation:.0f}ms")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline test suite")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of ChromaDB index")
    parser.add_argument("--compare", action="store_true",
                        help="Run hybrid vs vector vs BM25 retrieval comparison")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Evaluate retrieval only (no LLM calls) — useful when API quota is exhausted")
    parser.add_argument("--model", default=MODEL,
                        help=f"Gemini model (default: {MODEL})")
    args = parser.parse_args()

    print("Loading and chunking documents...")
    chunks = load_and_chunk_documents(DOCS_DIR)

    print("\nBuilding index...")
    collection, bm25, chunks = build_index(
        chunks,
        persist_dir=CHROMA_DIR,
        rebuild=args.rebuild,
    )

    if args.compare:
        # Demonstrate hybrid vs vector vs BM25 on a keyword-heavy query
        run_comparison(
            query="3.5 minimum review score seller suspension",
            collection=collection,
            bm25=bm25,
            chunks=chunks,
        )
        run_comparison(
            query="what happens if a seller has bad reviews",
            collection=collection,
            bm25=bm25,
            chunks=chunks,
        )

    results = run_rag_tests(
        collection, bm25, chunks, TEST_QUESTIONS,
        model=args.model,
        retrieval_only=args.retrieval_only,
    )
    print_summary(results)
