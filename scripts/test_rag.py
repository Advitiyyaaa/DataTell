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
    _has_sufficient_evidence,
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
        "evidence_terms": ["30 calendar days"],
    },
    {
        "id": 2,
        "question": "How long does standard delivery take to remote areas?",
        "expected_sources": ["shipping_policy.md"],
        "key_fact": "8 to 15 business days",
        "evidence_terms": ["8 to 15 business days"],
    },
    {
        "id": 3,
        "question": "What payment methods does Olist accept?",
        "expected_sources": ["customer_faq.md"],
        "key_fact": "credit card, boleto, debit card, PIX, vouchers",
        "evidence_terms": ["credit card", "boleto", "debit card", "pix", "vouchers"],
    },
    {
        "id": 4,
        "question": "What review score must sellers maintain to stay active on the platform?",
        "expected_sources": ["seller_guidelines.md"],
        "key_fact": "3.5 minimum",
        "evidence_terms": ["minimum acceptable", "3.5"],
    },
    {
        "id": 5,
        "question": "Can I cancel an order after it has already been shipped?",
        "expected_sources": ["customer_faq.md"],
        "key_fact": "No — once dispatched, cancellation is not possible",
        "evidence_terms": ["once dispatched", "cancellation is no longer possible"],
    },
    {
        "id": 6,
        "question": "What are the conditions for returning a purchased item?",
        "expected_sources": ["return_policy.md"],
        "key_fact": "unused, original packaging, within 7 days",
        "evidence_terms": ["unused", "original packaging", "7 calendar days"],
    },
    {
        "id": 7,
        "question": "Within how many business days must sellers dispatch orders?",
        "expected_sources": ["shipping_policy.md", "seller_guidelines.md"],
        "key_fact": "3 business days",
        "evidence_terms": ["3 business days"],
    },
    {
        "id": 8,
        "question": "What happens to sellers with consistently low review scores?",
        "expected_sources": ["seller_guidelines.md"],
        "key_fact": "suspension below 3.0 or below 3.5 for 60 days",
        "evidence_terms": ["below 3.0", "below 3.5", "60 consecutive days"],
    },
    {
        "id": 9,
        "question": "How is a boleto refund processed and how long does it take?",
        "expected_sources": ["return_policy.md"],
        "key_fact": "bank transfer within 5 business days",
        "evidence_terms": ["bank transfer", "5 business days"],
    },
    {
        "id": 10,
        "question": "What legal warranty period applies to electronics under Brazilian law?",
        "expected_sources": ["return_policy.md", "customer_faq.md"],
        "key_fact": "90 days",
        "evidence_terms": ["90 days"],
    },
    {
        "id": 11,
        "question": "Which airport offers same-day in-store pickup for Olist orders?",
        "expected_sources": [],
        "key_fact": "not answerable from the policy corpus",
        "evidence_terms": [],
        "expected_answerable": False,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _first_evidence_rank(retrieved: list[dict], question: dict) -> int | None:
    """Return the one-based rank of a chunk with the required source and evidence."""
    expected_sources = set(question["expected_sources"])
    evidence_terms = [_normalize(term) for term in question.get("evidence_terms", [])]
    for rank, chunk in enumerate(retrieved, 1):
        source_matches = chunk["metadata"].get("source") in expected_sources
        text = _normalize(chunk["text"])
        if source_matches and all(term in text for term in evidence_terms):
            return rank
    return None


def _answer_has_evidence(answer: str, question: dict) -> bool:
    evidence_terms = [_normalize(term) for term in question.get("evidence_terms", [])]
    return all(term in _normalize(answer) for term in evidence_terms)


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
            expected_answerable = q.get("expected_answerable", True)
            evidence_rank = _first_evidence_rank(retrieved, q) if expected_answerable else None
            evidence_hit = evidence_rank is not None
            answerable = _has_sufficient_evidence(retrieved, min_vector_score=0.25)
            passed = evidence_hit if expected_answerable else not answerable
            status = "[PASS]" if passed else "[MISS]"
            print(f"         {status} | Retrieved sources: {sources}")
            print(f"         Evidence rank   : {evidence_rank or 'not retrieved'}")
            if retrieved:
                print(f"         Top chunk: {retrieved[0]['text'][:120].replace(chr(10), ' ')}...\n")
            else:
                print("         Top chunk: none\n")
            results.append({
                "id": q["id"], "question": q["question"],
                "status": status, "source_hit": evidence_hit, "fact_hit": False,
                "evidence_hit": evidence_hit, "evidence_rank": evidence_rank,
                "answerable": answerable,
                "expected_answerable": expected_answerable,
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
                "evidence_hit": False,
                "evidence_rank": None,
                "answerable": None,
                "expected_answerable": q.get("expected_answerable", True),
                "sources": [],
                "retrieval_ms": 0,
                "generation_ms": 0,
                "answer_snippet": str(exc)[:200],
            })
            print()
            time.sleep(15)
            continue

        expected_answerable = q.get("expected_answerable", True)
        evidence_rank = _first_evidence_rank(result["chunks"], q) if expected_answerable else None
        evidence_hit = evidence_rank is not None
        fact_hit = _answer_has_evidence(result["answer"], q) if expected_answerable else not result["answerable"]
        if expected_answerable:
            status = "[PASS]" if (evidence_hit and fact_hit) else "[PARTIAL]" if evidence_hit else "[MISS]"
        else:
            status = "[PASS]" if not result["answerable"] else "[MISS]"

        print(f"         {status} | Sources: {result['sources']}")
        print(f"         Evidence rank: {evidence_rank or 'not retrieved'} | Answerable: {result['answerable']}")
        print(f"         Timing: retrieval={result['retrieval_ms']}ms | generation={result['generation_ms']}ms")
        print(f"\n         Answer:\n         {result['answer'][:400]}{'...' if len(result['answer']) > 400 else ''}\n")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "status": status,
            "source_hit": evidence_hit,
            "fact_hit": fact_hit,
            "evidence_hit": evidence_hit,
            "evidence_rank": evidence_rank,
            "answerable": result["answerable"],
            "expected_answerable": expected_answerable,
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
    answerable_results = [
        result for result in results
        if result.get("expected_answerable", True) and result["status"] != "[ERROR]"
    ]
    evidence_hits = sum(result.get("evidence_hit", False) for result in answerable_results)
    reciprocal_ranks = [
        1 / result["evidence_rank"]
        for result in answerable_results
        if result.get("evidence_rank")
    ]
    print(
        f"  Evidence Recall@4: {evidence_hits/max(len(answerable_results), 1)*100:.0f}% "
        f"({evidence_hits}/{len(answerable_results)})"
    )
    print(
        f"  Evidence MRR@4   : "
        f"{sum(reciprocal_ranks)/max(len(answerable_results), 1):.3f}"
    )
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
