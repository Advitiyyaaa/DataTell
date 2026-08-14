"""
migrate_to_turso.py — Phase 8: One-shot migration of analytics.db → Turso Cloud
================================================================================

Run this ONCE after setting up your Turso account and adding credentials to .env.

Usage:
    # Dry-run: verify counts only (no writes to Turso)
    python scripts/migrate_to_turso.py --dry-run

    # Full migration (upload all rows to Turso)
    python scripts/migrate_to_turso.py

Prerequisites:
    pip install requests        (likely already installed)
    TURSO_DATABASE_URL=libsql://your-db.turso.io  in .env
    TURSO_AUTH_TOKEN=your-token                    in .env

How it works:
    Uses Turso's HTTP REST API directly (no pyturso/Rust compilation needed).
    Batches INSERT statements in groups of 100 rows per request.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent
load_dotenv(dotenv_path=root_dir / ".env")

import os
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL")
TURSO_AUTH_TOKEN   = os.getenv("TURSO_AUTH_TOKEN")

LOCAL_DB = root_dir / "db" / "analytics.db"
BATCH_SIZE = 100


def _get_tables(conn: sqlite3.Connection) -> list[str]:
    """Return all user tables in the SQLite database."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _get_create_statement(conn: sqlite3.Connection, table: str) -> str:
    """Return the CREATE TABLE statement for a given table."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row[0]


def _row_count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cursor = conn.execute(f"SELECT * FROM {table} LIMIT 0")
    return [desc[0] for desc in cursor.description]


def migrate(dry_run: bool = False) -> None:
    # ── 1. Connect to local DB ────────────────────────────────────────────────
    if not LOCAL_DB.exists():
        sys.exit(f"ERROR: Local database not found at {LOCAL_DB}")

    print(f"Opening local database: {LOCAL_DB}")
    local_conn = sqlite3.connect(str(LOCAL_DB))
    tables = _get_tables(local_conn)
    print(f"Found {len(tables)} tables: {', '.join(tables)}\n")

    # ── 2. Verify env vars ────────────────────────────────────────────────────
    if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
        sys.exit(
            "ERROR: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in .env\n"
            "Sign up at https://turso.tech and create a database."
        )

    if dry_run:
        print("DRY-RUN MODE — counting local rows only (no Turso writes)\n")
        total = 0
        for table in tables:
            count = _row_count(local_conn, table)
            total += count
            print(f"  {table}: {count:,} rows")
        print(f"\nTotal rows to migrate: {total:,}")
        local_conn.close()
        return

    # ── 3. Connect to Turso via HTTP ──────────────────────────────────────────
    sys.path.insert(0, str(root_dir / "scripts"))
    import turso_client

    print(f"Connecting to Turso: {TURSO_DATABASE_URL}")
    turso_conn = turso_client.connect(
        remote_url=TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN,
    )
    print("Connected.\n")

    # ── 4. Create tables in Turso ─────────────────────────────────────────────
    for table in tables:
        create_sql = _get_create_statement(local_conn, table)
        # Make idempotent
        create_sql_safe = create_sql.replace(
            "CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1
        )
        print(f"Creating table '{table}'...")
        turso_conn.execute(create_sql_safe)

    print()

    # ── 5. Migrate rows ───────────────────────────────────────────────────────
    overall_start = time.time()
    for table in tables:
        local_total = _row_count(local_conn, table)
        try:
            turso_current = turso_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception:
            turso_current = 0

        if turso_current == local_total:
            print(f"Skipping '{table}' — already fully migrated ({turso_current:,} rows)")
            continue

        columns = _get_columns(local_conn, table)
        col_list = ", ".join(columns)
        col_count = len(columns)
        row_placeholder = f"({', '.join(['?'] * col_count)})"

        print(f"Migrating '{table}': {local_total:,} rows (currently {turso_current:,} in Turso)...")

        rows_inserted = 0
        t_start = time.time()

        # Batch size 50 creates ~150-350 params per statement, perfect for libSQL HTTP
        BATCH = 50
        cursor = local_conn.execute(f"SELECT {col_list} FROM {table}")
        batch = cursor.fetchmany(BATCH)

        while batch:
            multi_placeholders = ", ".join([row_placeholder] * len(batch))
            multi_sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES {multi_placeholders}"
            flat_params = [val for row in batch for val in row]

            # Retry loop with backoff for network resilience
            for attempt in range(1, 6):
                try:
                    turso_conn.execute(multi_sql, flat_params)
                    break
                except Exception as exc:
                    if attempt == 5:
                        raise
                    wait = attempt * 1.5
                    print(f"\n  [Retry {attempt}/5] Network error: {exc}. Retrying in {wait:.1f}s...")
                    time.sleep(wait)

            rows_inserted += len(batch)

            # Progress indicator
            pct = rows_inserted / local_total * 100 if local_total else 100
            elapsed = time.time() - t_start
            rate = rows_inserted / elapsed if elapsed > 0 else 0
            print(f"\r  {rows_inserted:,}/{local_total:,} ({pct:.1f}%) - {elapsed:.1f}s ({rate:.0f} rows/s)", end="", flush=True)

            batch = cursor.fetchmany(BATCH)

        print(f"\r  [OK] {table}: {local_total:,} rows migrated in {time.time() - t_start:.1f}s")

    print(f"\nTotal migration time: {time.time() - overall_start:.1f}s")

    # ── 6. Verify counts ──────────────────────────────────────────────────────
    print("\nVerifying row counts...")
    all_match = True
    for table in tables:
        local_count = _row_count(local_conn, table)
        turso_count = turso_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        match = "MATCH" if local_count == turso_count else "MISMATCH"
        print(f"  {table}: local={local_count:,}  turso={turso_count:,}  [{match}]")
        if local_count != turso_count:
            all_match = False

    local_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate analytics.db → Turso Cloud")
    parser.add_argument("--dry-run", action="store_true", help="Count rows only, no writes")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)
