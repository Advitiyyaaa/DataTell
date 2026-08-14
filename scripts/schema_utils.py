"""
schema_utils.py — Phase 2
==========================
Extracts the live database schema from SQLite and formats it into a
prompt-ready string for injection into the LLM system prompt.

Two public functions:
    get_schema(conn)             -> dict   (machine-readable)
    format_schema_for_prompt(schema) -> str    (human/LLM-readable)

Design note: schema is extracted at call-time (not cached) so it always
reflects the current DB state. For Phase 4 the agent will call this once
at startup and pass the schema dict to every nl_to_sql call.
"""

import sqlite3
import pandas as pd
from typing import Any


def get_schema(conn: sqlite3.Connection) -> dict:
    """
    Extract full schema from a SQLite connection.

    Returns
    -------
    dict  keyed by table name:
        {
            "table_name": {
                "columns": [{"name": str, "type": str}, ...],
                "row_count": int,
                "sample_rows": [{"col": value, ...}, ...]   # 3 rows
            },
            ...
        }
    """
    schema: dict[str, Any] = {}

    # Get all user tables (exclude SQLite internal tables)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    table_names = [row[0] for row in cursor.fetchall()]

    for table_name in table_names:
        # Column info
        cols_cur = conn.execute(f"PRAGMA table_info([{table_name}])")
        columns = [
            {"name": row[1], "type": row[2] or "TEXT"}
            for row in cols_cur.fetchall()
        ]

        # Row count
        count_cur = conn.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        count_row = count_cur.fetchone()
        count = count_row[0] if count_row else 0

        # Sample rows — 3 rows, values truncated to 50 chars for long strings
        sample_cur = conn.execute(f"SELECT * FROM [{table_name}] LIMIT 3")
        sample_cols = [desc[0] for desc in sample_cur.description] if sample_cur.description else []
        sample_rows = []
        for row in sample_cur.fetchall():
            cleaned = {}
            for col, val in zip(sample_cols, row):
                if isinstance(val, str) and len(val) > 50:
                    val = val[:47] + "..."
                cleaned[col] = val
            sample_rows.append(cleaned)

        schema[table_name] = {
            "columns": columns,
            "row_count": int(count),
            "sample_rows": sample_rows,
        }

    return schema


def format_schema_for_prompt(schema: dict, include_sample_rows: bool = False) -> str:
    """
    Convert the schema dict into a compact, LLM-readable string.

    Format per table:
        TABLE: <name> | <row_count> rows
        COLUMNS: col1(TYPE), col2(TYPE), ...
        SAMPLE:  (only when include_sample_rows=True)
          col1=val, col2=val, ...
          col1=val, col2=val, ...
    """
    lines = []

    for table_name, info in schema.items():
        # Header
        lines.append(f"TABLE: {table_name} | {info['row_count']:,} rows")

        # Columns
        col_str = ", ".join(
            f"{c['name']}({c['type']})" for c in info["columns"]
        )
        lines.append(f"COLUMNS: {col_str}")

        # Raw sample values can include PII. Keep them opt-in for trusted local data.
        if include_sample_rows and info["sample_rows"]:
            lines.append("SAMPLE:")
            for row in info["sample_rows"]:
                row_str = ", ".join(f"{k}={v!r}" for k, v in row.items())
                lines.append(f"  {row_str}")

        lines.append("")  # blank line between tables

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path

    DB_PATH = Path("db/analytics.db")
    conn = sqlite3.connect(DB_PATH)
    schema = get_schema(conn)

    print(f"Extracted schema for {len(schema)} tables:\n")
    print(format_schema_for_prompt(schema))
    conn.close()
