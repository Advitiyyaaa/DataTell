"""
Phase 1 – Data Foundation: Load & Clean
========================================
Loads Olist e-commerce CSVs into a SQLite database with documented cleaning steps.

Architecture note
-----------------
The core function is:  load_data(source_config: dict) -> sqlite3.Connection
All dataset-specific logic is expressed through `source_config`, not hardcoded.
This makes the loader swappable in Phase 9 ("bring your own CSV") without
touching any downstream code (NL→SQL, agent, UI).

Usage (from project root):
    python scripts/load_data.py

source_config schema:
    {
        "db_path": str | Path,          # where to write the SQLite file
        "tables": {
            "<table_name>": {
                "path": str | Path,     # CSV path
                "date_cols": [...],     # columns to parse as datetime
                "pk": str               # primary key column (for dup checks)
            },
            ...
        },
        "transforms": [callable],       # list of (dataframes_dict) -> dataframes_dict functions
    }
"""

import os
import sqlite3
import uuid
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Core loader — source_config-driven, nothing hardcoded
# ---------------------------------------------------------------------------

def _key_columns(cfg: dict) -> list[str]:
    """Return the configured unique key, including composite keys."""
    if cfg.get("key_columns"):
        return list(cfg["key_columns"])
    return [cfg["pk"]] if cfg.get("pk") else []


def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier after it has been validated against config."""
    return "[" + identifier.replace("]", "]]" ) + "]"


def _validate_relationships(frames: dict[str, pd.DataFrame], relationships: list[dict]) -> None:
    """Raise if a configured child key has no corresponding parent key."""
    for relationship in relationships:
        child_table = relationship["child_table"]
        child_column = relationship["child_column"]
        parent_table = relationship["parent_table"]
        parent_column = relationship["parent_column"]

        child_values = frames[child_table][child_column].dropna()
        parent_values = set(frames[parent_table][parent_column].dropna())
        orphan_count = int((~child_values.isin(parent_values)).sum())
        if orphan_count:
            raise ValueError(
                f"Referential-integrity check failed: {orphan_count:,} values in "
                f"{child_table}.{child_column} have no match in "
                f"{parent_table}.{parent_column}."
            )


def _create_configured_indexes(
    conn: sqlite3.Connection,
    frames: dict[str, pd.DataFrame],
    source_config: dict,
) -> None:
    """Create indexes declared in config after all tables have been loaded."""
    for table_name, cfg in source_config["tables"].items():
        if table_name not in frames:
            continue  # Helper tables may have been consumed by a transform.

        for columns in cfg.get("indexes", []):
            if isinstance(columns, str):
                columns = [columns]
            if not columns or any(col not in frames[table_name].columns for col in columns):
                raise ValueError(f"Invalid index configuration for table '{table_name}': {columns}")

            index_name = f"idx_{table_name}_{'_'.join(columns)}"
            quoted_columns = ", ".join(_quote_identifier(col) for col in columns)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {_quote_identifier(index_name)} "
                f"ON {_quote_identifier(table_name)} ({quoted_columns})"
            )


def load_data(source_config: dict) -> sqlite3.Connection:
    """
    Load, clean, and persist tabular data to SQLite.

    Parameters
    ----------
    source_config : dict
        Controls which CSVs to load, which columns are dates, and what
        post-load transforms to apply.  See module docstring for schema.

    Returns
    -------
    sqlite3.Connection
        Open connection to the written database (caller is responsible for
        closing it, or use as a context manager).
    """
    db_path = Path(source_config["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load raw CSVs into DataFrames
    print("Loading CSVs...")
    frames: dict[str, pd.DataFrame] = {}
    for table_name, cfg in source_config["tables"].items():
        csv_path = Path(cfg["path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing source file for '{table_name}': {csv_path}")

        df = pd.read_csv(csv_path)

        required_columns = set(cfg.get("required_columns", []))
        required_columns.update(cfg.get("date_cols", []))
        required_columns.update(_key_columns(cfg))
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            raise ValueError(
                f"Source '{table_name}' is missing configured columns: {missing_columns}"
            )

        # Parse date columns (they arrive as strings from CSV)
        for col in cfg.get("date_cols", []):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        frames[table_name] = df
        print(f"  Loaded {table_name}: {len(df):,} rows")

    # 2. Drop rows missing configured unique keys and remove duplicate keys.
    print("\nCleaning...")
    for table_name, cfg in source_config["tables"].items():
        key_columns = _key_columns(cfg)
        if not key_columns:
            continue

        before = len(frames[table_name])
        frames[table_name] = frames[table_name].dropna(subset=key_columns)
        dropped = before - len(frames[table_name])
        if dropped:
            print(f"  {table_name}: dropped {dropped} rows with null key {key_columns}")

        duplicates = int(frames[table_name].duplicated(subset=key_columns).sum())
        if duplicates:
            print(
                f"  {table_name}.{key_columns}: {duplicates} duplicate keys found "
                "— deduplicating (keeping first)"
            )
            frames[table_name] = frames[table_name].drop_duplicates(
                subset=key_columns, keep="first"
            )

    # 3. Audit nulls — document instead of blindly dropping.
    print("\nNull counts per table:")
    for name, df in frames.items():
        nulls = {col: int(n) for col, n in df.isnull().sum().items() if n > 0}
        print(f"  {name}: {nulls if nulls else 'no nulls'}")

    # 4. Apply dataset-specific transforms after raw-key validation.
    print("\nApplying transforms...")
    for transform_fn in source_config.get("transforms", []):
        frames = transform_fn(frames)

    # 5. Verify configured relationships before publishing the database.
    print("\nReferential-integrity checks...")
    _validate_relationships(frames, source_config.get("relationships", []))
    print("  All configured relationships are valid")

    # 6. Write a temporary database, then atomically publish it only on success.
    temp_db_path = db_path.with_name(
        f".{db_path.stem}.{uuid.uuid4().hex}.tmp{db_path.suffix}"
    )
    conn: sqlite3.Connection | None = None
    try:
        print(f"\nWriting temporary SQLite database -> {temp_db_path.name}")
        conn = sqlite3.connect(temp_db_path)
        for table_name, df in frames.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"  Wrote {table_name}: {len(df):,} rows")

        _create_configured_indexes(conn, frames, source_config)
        conn.commit()
    except Exception:
        if conn is not None:
            conn.close()
        if temp_db_path.exists():
            temp_db_path.unlink()
        raise
    else:
        conn.close()

    os.replace(temp_db_path, db_path)
    print(f"\nDone. Database atomically written to {db_path}")
    return sqlite3.connect(db_path)


# ---------------------------------------------------------------------------
# Olist-specific transform functions
# ---------------------------------------------------------------------------

def _translate_product_categories(frames: dict) -> dict:
    """
    Merge the English category translation into the products table.
    Decision: keep the Portuguese name as fallback so no row loses its category.
    610 products have no English translation (niche categories not in the
    translation CSV) — we fall back to the original Portuguese name.
    """
    if "products" in frames and "category_translation" in frames:
        products = frames["products"].merge(
            frames["category_translation"],
            on="product_category_name",
            how="left"
        )
        # Fallback: if no English translation exists, use original
        products["product_category_name_english"] = (
            products["product_category_name_english"]
            .fillna(products["product_category_name"])
        )
        frames["products"] = products
        # Drop the helper table — it was only needed for the merge
        del frames["category_translation"]
    return frames


# ---------------------------------------------------------------------------
# Olist source_config — swap this out for any other dataset in Phase 9
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")

OLIST_SOURCE_CONFIG = {
    "db_path": "db/analytics.db",
    "tables": {
        "customers": {
            "path": RAW_DIR / "olist_customers_dataset.csv",
            "pk": "customer_id",
            "date_cols": [],
            "indexes": [["customer_id"], ["customer_unique_id"]],
        },
        "orders": {
            "path": RAW_DIR / "olist_orders_dataset.csv",
            "pk": "order_id",
            "date_cols": [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
            "indexes": [["order_id"], ["customer_id"], ["order_purchase_timestamp"]],
        },
        "order_items": {
            "path": RAW_DIR / "olist_order_items_dataset.csv",
            "pk": None,
            "key_columns": ["order_id", "order_item_id"],
            "date_cols": ["shipping_limit_date"],
            "indexes": [["order_id"], ["product_id"], ["seller_id"]],
        },
        "payments": {
            "path": RAW_DIR / "olist_order_payments_dataset.csv",
            "pk": None,
            "key_columns": ["order_id", "payment_sequential"],
            "date_cols": [],
            "indexes": [["order_id"], ["payment_type"]],
        },
        "reviews": {
            "path": RAW_DIR / "olist_order_reviews_dataset.csv",
            "pk": "review_id",
            "date_cols": ["review_creation_date", "review_answer_timestamp"],
            "indexes": [["review_id"], ["order_id"]],
        },
        "products": {
            "path": RAW_DIR / "olist_products_dataset.csv",
            "pk": "product_id",
            "date_cols": [],
            "indexes": [["product_id"], ["product_category_name_english"]],
        },
        "sellers": {
            "path": RAW_DIR / "olist_sellers_dataset.csv",
            "pk": "seller_id",
            "date_cols": [],
            "indexes": [["seller_id"]],
        },
        # Helper table — consumed by transform, not written to DB
        "category_translation": {
            "path": RAW_DIR / "product_category_name_translation.csv",
            "pk": "product_category_name",
            "date_cols": [],
        },
    },
    "transforms": [_translate_product_categories],
    "relationships": [
        {"child_table": "orders", "child_column": "customer_id", "parent_table": "customers", "parent_column": "customer_id"},
        {"child_table": "order_items", "child_column": "order_id", "parent_table": "orders", "parent_column": "order_id"},
        {"child_table": "order_items", "child_column": "product_id", "parent_table": "products", "parent_column": "product_id"},
        {"child_table": "order_items", "child_column": "seller_id", "parent_table": "sellers", "parent_column": "seller_id"},
        {"child_table": "reviews", "child_column": "order_id", "parent_table": "orders", "parent_column": "order_id"},
        {"child_table": "payments", "child_column": "order_id", "parent_table": "orders", "parent_column": "order_id"},
    ],
}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = load_data(OLIST_SOURCE_CONFIG)

    # Quick sanity check — confirm joins work
    print("\n--- Verification sample: orders joined to customers ---")
    sample = pd.read_sql("""
        SELECT o.order_id,
               o.order_status,
               c.customer_city,
               c.customer_state,
               o.order_purchase_timestamp
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LIMIT 5
    """, conn)
    print(sample.to_string(index=False))

    conn.close()
