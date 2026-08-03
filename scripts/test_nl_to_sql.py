"""
test_nl_to_sql.py — Phase 2
=============================
Runs 20 test questions through the NL→SQL engine and produces a
pass/fail report with the SQL generated and row counts.

Usage (from project root):
    python scripts/test_nl_to_sql.py

    # Run only a specific tier
    python scripts/test_nl_to_sql.py --tier easy
    python scripts/test_nl_to_sql.py --tier medium
    python scripts/test_nl_to_sql.py --tier hard

Output: console report + results saved to db/test_results.csv
"""

import sys
import sqlite3
import argparse
import csv
from pathlib import Path
from datetime import datetime

# Allow importing sibling scripts
sys.path.insert(0, str(Path(__file__).parent))

from schema_utils import get_schema
from nl_to_sql import nl_to_sql

# ---------------------------------------------------------------------------
# 20 Test questions — Easy / Medium / Hard
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    # ---- Easy (5) — single table or trivial joins ----
    {
        "id": 1,
        "tier": "easy",
        "question": "How many orders are there in total?",
        "expected_rows": 1,  # single count row
    },
    {
        "id": 2,
        "tier": "easy",
        "question": "What are the top 5 cities by number of customers?",
        "expected_rows": 5,
    },
    {
        "id": 3,
        "tier": "easy",
        "question": "What percentage of orders have been delivered?",
        "expected_rows": 1,
    },
    {
        "id": 4,
        "tier": "easy",
        "question": "How many unique sellers are there?",
        "expected_rows": 1,
    },
    {
        "id": 5,
        "tier": "easy",
        "question": "What is the distribution of payment types (count and total value per type)?",
        "expected_rows": None,  # don't check exact row count
    },

    # ---- Medium (10) — multi-table joins, group-by, having ----
    {
        "id": 6,
        "tier": "medium",
        "question": "What is the average order value (total price per order) across all orders?",
        "expected_rows": 1,
    },
    {
        "id": 7,
        "tier": "medium",
        "question": "What are the top 5 product categories by total revenue?",
        "expected_rows": 5,
    },
    {
        "id": 8,
        "tier": "medium",
        "question": "Which Brazilian state has the highest number of orders?",
        "expected_rows": 1,
    },
    {
        "id": 9,
        "tier": "medium",
        "question": "What is the average review score for each product category? Show the top 10 categories by average score.",
        "expected_rows": 10,
    },
    {
        "id": 10,
        "tier": "medium",
        "question": "How many orders were placed each month in 2018? Show month and count, ordered chronologically.",
        "expected_rows": None,
    },
    {
        "id": 11,
        "tier": "medium",
        "question": "What is the average delivery time in days for delivered orders (from purchase to delivery)?",
        "expected_rows": 1,
    },
    {
        "id": 12,
        "tier": "medium",
        "question": "Which 5 sellers have the highest average review score among sellers with at least 10 reviews?",
        "expected_rows": 5,
    },
    {
        "id": 13,
        "tier": "medium",
        "question": "What is the average number of payment installments for credit card payments?",
        "expected_rows": 1,
    },
    {
        "id": 14,
        "tier": "medium",
        "question": "How many orders contain more than one item?",
        "expected_rows": 1,
    },
    {
        "id": 15,
        "tier": "medium",
        "question": "What percentage of orders used more than one payment method?",
        "expected_rows": 1,
    },

    # ---- Hard (5) — complex joins, subqueries, multi-step logic ----
    {
        "id": 16,
        "tier": "hard",
        "question": "What are the 3 product categories with the lowest average review score (minimum 50 reviews each)?",
        "expected_rows": 3,
    },
    {
        "id": 17,
        "tier": "hard",
        "question": "Which product category has the highest average freight cost as a percentage of item price?",
        "expected_rows": 1,
    },
    {
        "id": 18,
        "tier": "hard",
        "question": "How many customers placed more than one order? What percentage of total customers is that?",
        "expected_rows": 1,
    },
    {
        "id": 19,
        "tier": "hard",
        "question": "Show the monthly order count and cumulative order count for 2018, ordered by month.",
        "expected_rows": None,
    },
    {
        "id": 20,
        "tier": "hard",
        "question": "Which 5 sellers sold items from the most distinct product categories? Show seller_id, seller_city, and category count.",
        "expected_rows": 5,
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests(
    questions: list[dict],
    schema: dict,
    conn: sqlite3.Connection,
    model: str = "gemini-1.5-flash",
    verbose: bool = False,
) -> list[dict]:
    """Run all test questions and return result records."""
    results = []
    total = len(questions)

    print(f"\n{'='*70}")
    print(f"  DataTell Phase 2 -- NL->SQL Test Suite")
    print(f"  Model: {model} | Questions: {total}")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions, 1):
        tier_badge = {"easy": "[E]", "medium": "[M]", "hard": "[H]"}.get(q["tier"], "[?]")
        print(f"[{i:02d}/{total}] {tier_badge} [{q['tier'].upper():6s}] {q['question']}")

        result = nl_to_sql(
            question=q["question"],
            schema=schema,
            conn=conn,
            model=model,
            max_retries=3,
            verbose=verbose,
        )

        row_count = len(result["results"]) if result["success"] and result["results"] is not None else None
        status = "[PASS]" if result["success"] else "[FAIL]"

        print(f"         {status} | Attempts: {result['attempts']} | Rows returned: {row_count}")

        if result["success"] and result["results"] is not None and not result["results"].empty:
            # Show first 3 rows inline
            preview = result["results"].head(3)
            preview_str = preview.to_string(index=False)
            # Indent each line
            for line in preview_str.split("\n"):
                print(f"         {line}")

        if not result["success"]:
            print(f"         Error: {result['error']}")
            if result["sql"]:
                print(f"         Last SQL: {result['sql'][:120]}...")

        print()

        results.append({
            "id": q["id"],
            "tier": q["tier"],
            "question": q["question"],
            "success": result["success"],
            "attempts": result["attempts"],
            "row_count": row_count,
            "sql": result["sql"] or "",
            "error": result["error"] or "",
        })

    return results


def print_summary(results: list[dict]) -> None:
    """Print pass/fail summary broken down by tier."""
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed
    avg_attempts = sum(r["attempts"] for r in results) / total if total else 0

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total   : {total}")
    print(f"  Passed  : {passed} ({passed/total*100:.0f}%)")
    print(f"  Failed  : {failed}")
    print(f"  Avg attempts per question: {avg_attempts:.2f}")
    print()

    for tier in ["easy", "medium", "hard"]:
        tier_results = [r for r in results if r["tier"] == tier]
        tier_pass = sum(1 for r in tier_results if r["success"])
        tier_total = len(tier_results)
        badge = {"easy": "[E]", "medium": "[M]", "hard": "[H]"}[tier]
        print(f"  {badge} {tier.upper():6s}: {tier_pass}/{tier_total} passed")

    print(f"{'='*70}\n")

    if failed > 0:
        print("  Failed questions:")
        for r in results:
            if not r["success"]:
                print(f"    [{r['id']:02d}] {r['question'][:70]}")
                print(f"         Error: {r['error'][:100]}")
        print()


def save_results(results: list[dict], out_path: Path) -> None:
    """Save results to CSV for future reference."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "tier", "question", "success", "attempts", "row_count", "sql", "error"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NL→SQL test suite")
    parser.add_argument(
        "--tier",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which tier of questions to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gemini-3.5-flash",
        help="Gemini model to use (default: gemini-3.5-flash)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generated SQL for each question",
    )
    args = parser.parse_args()

    DB_PATH = Path(__file__).parent.parent / "db" / "analytics.db"
    OUT_PATH = Path(__file__).parent.parent / "db" / "test_results.csv"

    conn = sqlite3.connect(DB_PATH)
    schema = get_schema(conn)

    questions = TEST_QUESTIONS if args.tier == "all" else [
        q for q in TEST_QUESTIONS if q["tier"] == args.tier
    ]

    results = run_tests(questions, schema, conn, model=args.model, verbose=args.verbose)
    print_summary(results)
    save_results(results, OUT_PATH)

    conn.close()
