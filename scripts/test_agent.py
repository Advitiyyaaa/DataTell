"""
test_agent.py -- Phase 4
========================
Test harness for the agent router.

Runs 23 mixed questions and reports:
  - Per-question: route predicted, expected route, pass/fail
  - Overall routing accuracy
  - Timing breakdown by route type
  - Summary table

Usage (from project root):
    python scripts/test_agent.py              # full suite (slow, hits API)
    python scripts/test_agent.py --fast       # skip BOTH questions (no parallel)
    python scripts/test_agent.py --q <index>  # run single question by index
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from agent_router import load_resources, build_agent_graph, run_agent

# ---------------------------------------------------------------------------
# Test question bank
# ---------------------------------------------------------------------------

# Each entry: (question, expected_route, description)
TEST_QUESTIONS = [
    # ------------------------------------------------------------------
    # SQL -- pure database queries (8 questions)
    # ------------------------------------------------------------------
    (
        "What are the top 5 product categories by total revenue?",
        "SQL",
        "SQL-01: Top categories by revenue",
    ),
    (
        "Which Brazilian state had the most customer orders in 2017?",
        "SQL",
        "SQL-02: State order count 2017",
    ),
    (
        "What is the average delivery time in days for orders placed in Sao Paulo?",
        "SQL",
        "SQL-03: Avg delivery time SP",
    ),
    (
        "How many orders were delivered after the estimated delivery date?",
        "SQL",
        "SQL-04: Late delivery count",
    ),
    (
        "Which 5 sellers have the highest average review scores?",
        "SQL",
        "SQL-05: Top sellers by review score",
    ),
    (
        "Show me the monthly order volume trend for 2017.",
        "SQL",
        "SQL-06: Monthly order trend 2017",
    ),
    (
        "What is the average payment value broken down by payment type?",
        "SQL",
        "SQL-07: Avg payment by type",
    ),
    (
        "Which product categories have the most 1-star reviews?",
        "SQL",
        "SQL-08: Categories with most 1-star reviews",
    ),
    # ------------------------------------------------------------------
    # RAG -- pure policy document queries (7 questions)
    # ------------------------------------------------------------------
    (
        "What is the return window for defective products?",
        "RAG",
        "RAG-01: Defective product return window",
    ),
    (
        "What are the seller performance requirements on Olist?",
        "RAG",
        "RAG-02: Seller performance requirements",
    ),
    (
        "How does the Olist shipping policy handle late deliveries?",
        "RAG",
        "RAG-03: Shipping policy for late deliveries",
    ),
    (
        "What is the maximum number of payment installments allowed?",
        "RAG",
        "RAG-04: Max payment installments",
    ),
    (
        "What happens to a seller who consistently receives low review scores?",
        "RAG",
        "RAG-05: Consequences of low review scores",
    ),
    (
        "How can a customer request a refund?",
        "RAG",
        "RAG-06: Customer refund process",
    ),
    (
        "What are the prohibited item categories on the Olist marketplace?",
        "RAG",
        "RAG-07: Prohibited item categories",
    ),
    # ------------------------------------------------------------------
    # BOTH -- compound queries requiring data + policy (5 questions)
    # ------------------------------------------------------------------
    (
        "What are the top 5 product categories by revenue, and what is the return policy for defective products?",
        "BOTH",
        "BOTH-01: Top categories + return policy",
    ),
    (
        "How many orders were delivered late in 2017, and what are the seller guidelines for handling shipping delays?",
        "BOTH",
        "BOTH-02: Late delivery count + seller delay guidelines",
    ),
    (
        "Which sellers have the highest average review scores, and what criteria does Olist use to evaluate seller performance?",
        "BOTH",
        "BOTH-03: Top sellers by score + evaluation criteria",
    ),
    (
        "What is the average order value by state, and what does the shipping policy say about regional delivery?",
        "BOTH",
        "BOTH-04: Avg order value by state + shipping policy",
    ),
    (
        "Which payment methods are most popular among customers, and what are the installment payment policies?",
        "BOTH",
        "BOTH-05: Payment method popularity + installment policy",
    ),
    # ------------------------------------------------------------------
    # CHITCHAT -- unanswerable / meta / greetings (3 questions)
    # ------------------------------------------------------------------
    (
        "Hello! What can you help me with?",
        "CHITCHAT",
        "CC-01: Greeting / capability question",
    ),
    (
        "What is the capital of France?",
        "CHITCHAT",
        "CC-02: Completely off-topic question",
    ),
    (
        "Who built this system and what data does it use?",
        "CHITCHAT",
        "CC-03: Meta-question about the system",
    ),
]


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def _route_emoji(route: str) -> str:
    return {"SQL": "DB", "RAG": "DOCS", "BOTH": "BOTH", "CHITCHAT": "CHAT"}.get(route, "?")


def _result_mark(predicted: str, expected: str) -> str:
    return "PASS" if predicted == expected else "FAIL"


def _print_separator(char: str = "-", width: int = 100) -> None:
    print(char * width)


def _print_question_result(
    idx: int,
    desc: str,
    question: str,
    expected: str,
    result: dict,
) -> None:
    predicted = result["route"]
    mark = _result_mark(predicted, expected)
    total_ms = result["total_ms"]

    print(f"\n[{idx:02d}] {desc}")
    print(f"  Q        : {question[:90]}{'...' if len(question) > 90 else ''}")
    print(f"  Expected : {_route_emoji(expected):<6}  |  Predicted : {_route_emoji(predicted):<6}  |  {mark}  |  {total_ms}ms")
    print(f"  Reason   : {result.get('route_reason', '')}")

    if result.get("error"):
        print(f"  ERROR    : {result['error']}")

    # SQL summary
    if result.get("sql_result"):
        s = result["sql_result"]
        status = "OK" if s.get("success") else "FAIL"
        print(f"  SQL      : [{status}] {s.get('row_count', 0)} rows | "
              f"{s.get('tool_ms', 0)}ms | attempts={s.get('attempts', 0)}")
        if not s.get("success"):
            print(f"  SQL ERR  : {s.get('error', '')[:120]}")

    # RAG summary
    if result.get("rag_result"):
        r = result["rag_result"]
        status = "OK" if r.get("answerable") else "NO ANSWER"
        print(f"  RAG      : [{status}] sources={r.get('sources', [])} | {r.get('tool_ms', 0)}ms")

    # Print answer preview
    answer = result.get("final_answer", "")
    if answer:
        lines = answer.strip().split("\n")
        preview = " | ".join(l.strip() for l in lines[:3] if l.strip())
        print(f"  Answer   : {preview[:150]}{'...' if len(preview) > 150 else ''}")


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(questions, graph, verbose: bool = True, delay_s: float = 5.0) -> dict:
    """Run test questions, return stats dict."""
    results = []
    correct = 0
    quota_exhausted = False

    for idx, (question, expected, desc) in enumerate(questions, 1):
        print(f"\n{'='*100}")
        print(f"Running [{idx}/{len(questions)}]: {desc}")
        print(f"{'='*100}")

        # Pause between questions to respect per-minute rate limits
        if idx > 1 and delay_s > 0:
            print(f"  [delay {delay_s:.0f}s between questions]")
            time.sleep(delay_s)

        try:
            result = run_agent(question, graph)
            predicted = result["route"]
            passed = predicted == expected
            if passed:
                correct += 1

            # Check if the run hit a daily quota error
            err = result.get("error", "") or ""
            if "Daily API quota exhausted" in err or "GenerateRequestsPerDayPerProjectPerModel" in err:
                quota_exhausted = True

            results.append({
                "idx":       idx,
                "desc":      desc,
                "question":  question,
                "expected":  expected,
                "predicted": predicted,
                "passed":    passed,
                "total_ms":  result["total_ms"],
                "result":    result,
            })

            if verbose:
                _print_question_result(idx, desc, question, expected, result)

        except Exception as exc:
            err_str = str(exc)
            print(f"  EXCEPTION: {err_str[:200]}")
            is_daily = (
                "Daily API quota" in err_str
                or "GenerateRequestsPerDayPerProjectPerModel" in err_str
                or "PerDay" in err_str
            )
            results.append({
                "idx":       idx,
                "desc":      desc,
                "question":  question,
                "expected":  expected,
                "predicted": "ERROR",
                "passed":    False,
                "total_ms":  0,
                "error":     err_str,
                "result":    None,
            })
            if is_daily:
                print(f"\n  *** Daily API quota exhausted at question {idx}/{len(questions)}. ***")
                print("  *** Stopping early. Remaining questions will be skipped. ***")
                print("  *** Run again tomorrow, or use --q N to test individual questions. ***\n")
                quota_exhausted = True
                break

    return {
        "results":          results,
        "total":            len(questions),
        "completed":        len(results),
        "correct":          correct,
        "accuracy":         correct / len(results) if results else 0.0,
        "quota_exhausted":  quota_exhausted,
    }


def print_summary(stats: dict) -> None:
    """Print a summary table of all results."""
    completed = stats.get("completed", len(stats["results"]))
    total     = stats["total"]
    stopped_early = stats.get("quota_exhausted", False) and completed < total

    print(f"\n\n{'#'*100}")
    title = f"TEST SUMMARY  ({completed}/{total} questions completed)"
    if stopped_early:
        title += "  [STOPPED EARLY — daily quota]"
    print(f"# {title}")
    print(f"{'#'*100}")

    results = stats["results"]

    # Summary table header
    print(f"\n{'IDX':<5} {'DESC':<35} {'EXPECTED':<10} {'PREDICTED':<10} {'RESULT':<6} {'MS':>6}")
    _print_separator()

    by_route: dict = {}
    for r in results:
        route = r["expected"]
        by_route.setdefault(route, {"total": 0, "correct": 0, "ms": []})
        by_route[route]["total"] += 1
        if r["passed"]:
            by_route[route]["correct"] += 1
        by_route[route]["ms"].append(r["total_ms"])

        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{r['idx']:<5} {r['desc'][:34]:<35} {r['expected']:<10} {r['predicted']:<10} {mark:<6} {r['total_ms']:>6}")

    _print_separator()

    # Per-route breakdown
    print("\nROUTE ACCURACY BREAKDOWN:")
    print(f"  {'Route':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>9} {'Avg ms':>8}")
    _print_separator("-", 55)
    for route in ["SQL", "RAG", "BOTH", "CHITCHAT"]:
        data = by_route.get(route, {"total": 0, "correct": 0, "ms": []})
        acc  = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        avg_ms = int(sum(data["ms"]) / len(data["ms"])) if data["ms"] else 0
        print(f"  {route:<12} {data['correct']:>8} {data['total']:>7} {acc:>8.1%} {avg_ms:>7}ms")

    _print_separator("-", 55)
    print(f"\n  OVERALL ACCURACY: {stats['correct']}/{stats['total']} "
          f"= {stats['accuracy']:.1%}")

    # List failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for r in failures:
            print(f"    [{r['idx']:02d}] {r['desc']}")
            print(f"         Expected {r['expected']} -> Got {r['predicted']}")
    else:
        print("\n  All tests passed!")

    # Goal check
    target = 0.85
    status = "PASS" if stats["accuracy"] >= target else "FAIL"
    print(f"\n  Phase 4 Target (>=85% accuracy): {status}")
    print(f"{'#'*100}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataTell Phase 4 Agent Test Suite")
    parser.add_argument("--fast",        action="store_true", help="Skip BOTH questions (faster)")
    parser.add_argument("--q",           type=int, default=None, metavar="N", help="Run only question N (1-indexed)")
    parser.add_argument("--no-summary",  action="store_true", help="Skip summary table")
    parser.add_argument("--delay",       type=float, default=5.0, metavar="SEC",
                        help="Seconds to wait between questions to respect RPM limits (default: 5)")
    args = parser.parse_args()

    # Load resources once
    print("Initializing agent resources...\n")
    conn, schema, collection, bm25, chunks = load_resources()
    graph = build_agent_graph(conn, schema, collection, bm25, chunks)
    print("\nAgent graph compiled. Starting tests...\n")

    # Filter questions
    questions = TEST_QUESTIONS

    if args.q is not None:
        idx = args.q - 1
        if 0 <= idx < len(TEST_QUESTIONS):
            questions = [TEST_QUESTIONS[idx]]
        else:
            print(f"Invalid question index {args.q}. Valid range: 1-{len(TEST_QUESTIONS)}")
            conn.close()
            sys.exit(1)
    elif args.fast:
        questions = [(q, r, d) for q, r, d in TEST_QUESTIONS if r != "BOTH"]
        print(f"Fast mode: skipping BOTH questions. Running {len(questions)} tests.\n")

    t0 = time.time()
    stats = run_tests(questions, graph, verbose=True, delay_s=args.delay)
    total_time = time.time() - t0

    if not args.no_summary:
        print_summary(stats)

    if stats.get("quota_exhausted"):
        completed = stats["completed"]
        total = stats["total"]
        print(f"\nNOTE: Suite stopped early at {completed}/{total} questions due to daily quota.")
        print(f"Resume tomorrow with: python scripts/test_agent.py --q N  (N = {completed+1} to {total})")

    print(f"Total wall-clock time: {total_time:.1f}s")
    conn.close()

