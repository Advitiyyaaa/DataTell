"""Phase 1 database verification with enforceable data-quality checks."""

import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).parent.parent / "db" / "analytics.db"

EXPECTED_ROW_COUNTS = {
    "customers": 99_441,
    "orders": 99_441,
    "order_items": 112_650,
    "payments": 103_886,
    "reviews": 98_410,
    "products": 32_951,
    "sellers": 3_095,
}

KEYS = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "payments": ["order_id", "payment_sequential"],
    "reviews": ["review_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
}

RELATIONSHIPS = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("order_items", "seller_id", "sellers", "seller_id"),
    ("reviews", "order_id", "orders", "order_id"),
    ("payments", "order_id", "orders", "order_id"),
]


def _quote_identifier(identifier: str) -> str:
    return "[" + identifier.replace("]", "]]" ) + "]"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(
        pd.read_sql(f"SELECT COUNT(*) AS cnt FROM {_quote_identifier(table_name)}", conn)
        .iloc[0, 0]
    )


def verify_database(conn: sqlite3.Connection) -> None:
    """Validate schema shape, keys, relationships, and analytical smoke queries."""
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )["name"].tolist()
    actual_tables = set(tables)
    expected_tables = set(EXPECTED_ROW_COUNTS)
    _assert(
        actual_tables == expected_tables,
        f"Unexpected tables. Expected {sorted(expected_tables)}, got {sorted(actual_tables)}",
    )

    print("=== Row-count checks ===")
    for table_name, expected_count in EXPECTED_ROW_COUNTS.items():
        actual_count = _count(conn, table_name)
        _assert(
            actual_count == expected_count,
            f"{table_name}: expected {expected_count:,} rows, got {actual_count:,}",
        )
        print(f"  PASS {table_name}: {actual_count:,} rows")

    print("\n=== Key integrity checks ===")
    for table_name, key_columns in KEYS.items():
        quoted_columns = ", ".join(_quote_identifier(column) for column in key_columns)
        null_predicate = " OR ".join(
            f"{_quote_identifier(column)} IS NULL" for column in key_columns
        )
        null_count = int(
            pd.read_sql(
                f"SELECT COUNT(*) AS cnt FROM {_quote_identifier(table_name)} "
                f"WHERE {null_predicate}",
                conn,
            ).iloc[0, 0]
        )
        duplicate_count = int(
            pd.read_sql(
                f"SELECT COUNT(*) AS cnt FROM ("
                f"SELECT {quoted_columns}, COUNT(*) AS n "
                f"FROM {_quote_identifier(table_name)} "
                f"GROUP BY {quoted_columns} HAVING n > 1"
                f")",
                conn,
            ).iloc[0, 0]
        )
        _assert(null_count == 0, f"{table_name}: {null_count} null key rows")
        _assert(duplicate_count == 0, f"{table_name}: {duplicate_count} duplicate keys")
        print(f"  PASS {table_name}: non-null, unique {key_columns}")

    print("\n=== Referential-integrity checks ===")
    for child_table, child_column, parent_table, parent_column in RELATIONSHIPS:
        orphan_count = int(
            pd.read_sql(
                f"SELECT COUNT(*) AS cnt "
                f"FROM {_quote_identifier(child_table)} child "
                f"LEFT JOIN {_quote_identifier(parent_table)} parent "
                f"ON child.{_quote_identifier(child_column)} = parent.{_quote_identifier(parent_column)} "
                f"WHERE child.{_quote_identifier(child_column)} IS NOT NULL "
                f"AND parent.{_quote_identifier(parent_column)} IS NULL",
                conn,
            ).iloc[0, 0]
        )
        _assert(
            orphan_count == 0,
            f"{child_table}.{child_column}: {orphan_count} orphan foreign keys",
        )
        print(f"  PASS {child_table}.{child_column} -> {parent_table}.{parent_column}")

    print("\n=== Analytical smoke checks ===")
    checks = {
        "order status breakdown": """
            SELECT order_status, COUNT(*) AS order_count
            FROM orders
            GROUP BY order_status
        """,
        "category revenue": """
            SELECT p.product_category_name_english, SUM(oi.price) AS total_revenue
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            WHERE p.product_category_name_english IS NOT NULL
            GROUP BY p.product_category_name_english
            ORDER BY total_revenue DESC
            LIMIT 5
        """,
        "payment breakdown": """
            SELECT payment_type, COUNT(*) AS payment_count
            FROM payments
            GROUP BY payment_type
        """,
    }
    for name, query in checks.items():
        result = pd.read_sql(query, conn)
        _assert(not result.empty, f"Analytical smoke check returned no rows: {name}")
        print(f"  PASS {name}: {len(result)} rows")


if __name__ == "__main__":
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}. Run load_data.py first.")

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        verify_database(conn)
    finally:
        conn.close()

    print("\nAll database verification checks passed.")
