"""
nl_to_sql.py — Phase 2
========================
Core NL→SQL engine.

Public API:
    nl_to_sql(question, schema, conn, ...) -> dict

The function:
  1. Builds a system prompt with the full injected schema
  2. Calls Gemini to generate SQL
  3. Extracts + validates SQL (SELECT-only safety check)
  4. Executes against SQLite
  5. On failure, sends the error back to the model and retries (max 3 attempts)
  6. Returns a structured result dict

Usage (from project root):
    python scripts/nl_to_sql.py "What are the top 5 cities by number of orders?"
"""

import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from project root regardless of cwd
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

_api_key = os.getenv("GEMINI_API_KEY")


# ---------------------------------------------------------------------------
# System prompt — injected schema goes at the bottom
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """You are an expert SQL analyst working with a SQLite database containing \
Brazilian e-commerce data from the Olist platform (2016-2018).

YOUR TASK: Given a natural language question, generate a single SQL SELECT query that answers it.

CRITICAL RULES — follow every one:
1. Return ONLY the raw SQL query. No explanations, no markdown fences, no comments, no preamble.
2. The query MUST be a single read-only SELECT statement. It may start with SELECT or WITH. NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, TRUNCATE, ATTACH, or PRAGMA.
3. Use standard SQLite syntax only.
4. For date operations use SQLite functions:
   - Extract year:  strftime('%Y', col)
   - Extract month: strftime('%Y-%m', col)
   - Date diff:     CAST((julianday(end_col) - julianday(start_col)) AS INTEGER)
   - NOT: EXTRACT(), DATE_TRUNC(), DATEDIFF(), INTERVAL
5. Use LOWER() for case-insensitive text matching. NOT ILIKE.
6. Always use table aliases (e.g. FROM orders o JOIN customers c ON ...).
7. For revenue calculations use order_items.price (not payments.payment_value — payments can be split across installments).
8. The `reviews` table has review_comment_message (free text) and review_score (integer 1-5).
9. For ANY question about products or categories (top/best/worst/most/least/all/any), ALWAYS group by and SELECT p.product_category_name_english (not product_id). Join order_items with products ON oi.product_id = p.product_id. Rank by aggregate (COUNT, SUM, AVG) and apply LIMIT. Add HAVING COUNT(*) >= 10 when computing averages to ensure statistical significance. NEVER return raw product_id UUIDs as the primary display column.
10. product_id is a UUID and MUST NEVER appear as the sole identifying column in results shown to users. If a query involves the products table, you MUST SELECT p.product_category_name_english as the human-readable label.
11. STRICTLY FORBIDDEN: SELECT product_id FROM ... without also selecting p.product_category_name_english. Every product result MUST include the category name.

TABLE RELATIONSHIPS (foreign keys — not enforced in SQLite but must be respected):
  orders.customer_id        → customers.customer_id
  order_items.order_id      → orders.order_id
  order_items.product_id    → products.product_id
  order_items.seller_id     → sellers.seller_id
  reviews.order_id          → orders.order_id
  payments.order_id         → orders.order_id

DATABASE SCHEMA:
{schema}
"""

_RETRY_PROMPT_TEMPLATE = """\
The SQL query you generated failed to execute. Here is what went wrong:

FAILED SQL:
{sql}

ERROR:
{error}

Please generate a corrected SQL query. Return ONLY the raw SQL — no explanations, no markdown.
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_system_prompt(schema_str: str) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(schema=schema_str)


def extract_sql_from_response(response_text: str) -> str:
    """
    Strip markdown code fences if the LLM wraps the SQL in them.
    Returns the raw SQL string.
    """
    text = response_text.strip()

    # Match ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return text


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments before structural validation."""
    without_line_comments = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"/\*.*?\*/", "", without_line_comments, flags=re.DOTALL)


def validate_select_only(sql: str) -> None:
    """
    Raise ValueError if the SQL is not a pure SELECT statement.

    Safety layer: prevents the LLM from generating write/DDL operations.
    We strip comments first so tricks like '-- SELECT\\nDROP TABLE' don't slip through.
    """
    clean = _strip_sql_comments(sql).strip()
    if clean.endswith(";"):
        clean = clean[:-1].rstrip()
    if not clean:
        raise ValueError("Expected a SQL query, got an empty response.")
    if ";" in clean:
        raise ValueError("Safety violation: multiple SQL statements are not permitted.")

    match = re.match(r"([A-Za-z]+)", clean)
    first_word = match.group(1).upper() if match else ""

    FORBIDDEN = {
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
        "TRUNCATE", "REPLACE", "MERGE", "UPSERT", "EXEC", "EXECUTE",
        "GRANT", "REVOKE", "ATTACH", "DETACH", "PRAGMA",
    }

    if first_word in FORBIDDEN:
        raise ValueError(
            f"Safety violation: query starts with '{first_word}'. "
            "Only read-only SELECT statements are permitted."
        )
    if first_word not in {"SELECT", "WITH"}:
        raise ValueError(
            f"Expected query to start with SELECT or WITH, got '{first_word}'."
        )

    # A data-modifying CTE can begin with WITH, so reject forbidden keywords
    # anywhere outside simple quoted string literals.
    literal_free = re.sub(r"'(?:''|[^'])*'", "''", clean)
    keywords = set(re.findall(r"\b[A-Za-z]+\b", literal_free.upper()))
    disallowed = keywords & FORBIDDEN
    if disallowed:
        raise ValueError(
            "Safety violation: query contains forbidden keyword(s): "
            f"{', '.join(sorted(disallowed))}."
        )


_SQLITE_WRITE_ACTIONS = {
    getattr(sqlite3, name)
    for name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW",
        "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE", "SQLITE_ATTACH",
        "SQLITE_DETACH", "SQLITE_PRAGMA", "SQLITE_TRANSACTION",
    )
    if hasattr(sqlite3, name)
}


def _read_only_authorizer(action_code, _arg1, _arg2, _database, _source):
    """SQLite's final enforcement layer for generated SQL."""
    return sqlite3.SQLITE_DENY if action_code in _SQLITE_WRITE_ACTIONS else sqlite3.SQLITE_OK


def execute_sql(
    sql: str,
    conn: sqlite3.Connection,
    max_rows: int = 1_000,
    timeout_seconds: float = 5.0,
) -> pd.DataFrame:
    """Execute validated SQL with read-only authorization, timeout, and row cap."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    query = sql.strip().rstrip(";")
    guarded_sql = f"SELECT * FROM ({query}) AS generated_query LIMIT {max_rows + 1}"
    deadline = time.monotonic() + timeout_seconds

    def abort_if_timed_out() -> int:
        return int(time.monotonic() >= deadline)

    conn.set_authorizer(_read_only_authorizer)
    conn.set_progress_handler(abort_if_timed_out, 10_000)
    try:
        results = pd.read_sql(guarded_sql, conn)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError(f"SQL query exceeded {timeout_seconds:.1f}s timeout") from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)
        conn.set_authorizer(None)

    truncated = len(results) > max_rows
    if truncated:
        results = results.iloc[:max_rows].copy()
    results.attrs["truncated"] = truncated
    return results


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def nl_to_sql(
    question: str,
    schema: dict,
    conn: sqlite3.Connection,
    model: str = "gemini-3.5-flash",
    max_retries: int = 3,
    max_rows: int = 1_000,
    timeout_seconds: float = 5.0,
    include_sample_rows: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Convert a natural language question to SQL and execute it.

    Parameters
    ----------
    question   : The natural language question to answer.
    schema     : Schema dict from schema_utils.get_schema().
    conn       : Open SQLite connection.
    model      : Gemini model name (default: gemini-2.5-flash).
    max_retries: Max LLM attempts before giving up (default: 3).
    max_rows   : Maximum number of rows returned to the caller.
    timeout_seconds: SQLite execution timeout enforced with a progress handler.
    include_sample_rows: Include raw data samples in the prompt. Keep False for PII safety.
    verbose    : Print each attempt's SQL and error if True.

    Returns
    -------
    {
        "question": str,
        "sql":      str | None,      # final SQL that was executed (or last attempted)
        "results":  pd.DataFrame | None,
        "attempts": int,             # how many LLM calls were made
        "success":  bool,
        "error":    str | None       # last error if all retries exhausted
    }
    """
    if not _api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=<your key>."
        )

    # Import here to keep schema_utils independent
    from schema_utils import format_schema_for_prompt

    schema_str = format_schema_for_prompt(schema, include_sample_rows=include_sample_rows)
    system_prompt = _build_system_prompt(schema_str)

    client = genai.Client(api_key=_api_key)

    # Build conversation history for multi-turn retry
    # google-genai uses contents list: alternating user/model turns
    history: list[types.Content] = []

    last_sql: Optional[str] = None
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        # Build the user message for this attempt
        if attempt == 1:
            user_text = f"Question: {question}"
        else:
            user_text = _RETRY_PROMPT_TEMPLATE.format(
                sql=last_sql or "(no SQL generated)",
                error=last_error or "(unknown error)",
            )

        if verbose:
            print(f"\n  [Attempt {attempt}/{max_retries}]")
            if attempt > 1:
                print(f"  Retrying after error: {last_error}")

        # Append user turn to history
        history.append(types.Content(
            role="user",
            parts=[types.Part(text=user_text)]
        ))

        try:
            # Inner rate-limit retry with exponential backoff
            # (separate from the SQL-error retry loop above)
            import time
            api_response = None
            for backoff_attempt in range(3):
                try:
                    api_response = client.models.generate_content(
                        model=model,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=0.0,
                        ),
                    )
                    break  # success — exit backoff loop
                except Exception as api_exc:
                    if "429" in str(api_exc) and backoff_attempt < 2:
                        wait_sec = 35 * (backoff_attempt + 1)
                        if verbose:
                            print(f"  Rate limited. Waiting {wait_sec}s before retry...")
                        time.sleep(wait_sec)
                    else:
                        raise  # non-rate-limit error or last backoff attempt

            raw_response = api_response.text
            raw_sql = extract_sql_from_response(raw_response)
            last_sql = raw_sql

            # Append model turn to history (for retry context)
            history.append(types.Content(
                role="model",
                parts=[types.Part(text=raw_response)]
            ))

            if verbose:
                print(f"  Generated SQL:\n    {raw_sql[:200]}{'...' if len(raw_sql) > 200 else ''}")

            # Safety check
            validate_select_only(raw_sql)

            # Execute in a read-only, time-limited sandbox.
            execution_started = time.monotonic()
            results = execute_sql(
                raw_sql,
                conn,
                max_rows=max_rows,
                timeout_seconds=timeout_seconds,
            )
            execution_ms = round((time.monotonic() - execution_started) * 1000)

            return {
                "question": question,
                "sql": raw_sql,
                "results": results,
                "attempts": attempt,
                "success": True,
                "error": None,
                "row_count": len(results),
                "truncated": bool(results.attrs.get("truncated", False)),
                "execution_ms": execution_ms,
            }

        except Exception as exc:
            last_error = str(exc)
            if verbose:
                print(f"  Error: {last_error}")
            # If the model turn wasn't appended yet (error before response), don't append
            # The next user message will contain the error context

    # All retries exhausted
    return {
        "question": question,
        "sql": last_sql,
        "results": None,
        "attempts": max_retries,
        "success": False,
        "error": last_error,
        "row_count": 0,
        "truncated": False,
        "execution_ms": None,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint — quick interactive test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from schema_utils import get_schema

    DB_PATH = Path(__file__).parent.parent / "db" / "analytics.db"
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What are the top 5 product categories by total revenue?"

    conn = sqlite3.connect(DB_PATH)
    schema = get_schema(conn)

    print(f"\nQuestion: {question}\n")
    result = nl_to_sql(question, schema, conn, verbose=True)
    print(f"\n{'='*60}")
    print(f"Success : {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"SQL     :\n{result['sql']}")
    if result["success"]:
        print(f"\nResults ({len(result['results'])} rows):")
        print(result["results"].to_string(index=False))
    else:
        print(f"Error   : {result['error']}")

    conn.close()
