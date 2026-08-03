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

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Core loader — source_config-driven, nothing hardcoded
# ---------------------------------------------------------------------------

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
        df = pd.read_csv(cfg["path"])

        # Parse date columns (they arrive as strings from CSV)
        for col in cfg.get("date_cols", []):
            df[col] = pd.to_datetime(df[col], errors="coerce")

        frames[table_name] = df
        print(f"  Loaded {table_name}: {len(df):,} rows")

    # 2. Apply dataset-specific transforms (merges, derivations, etc.)
    print("\nApplying transforms...")
    for transform_fn in source_config.get("transforms", []):
        frames = transform_fn(frames)

    # 3. Drop rows missing primary keys (data integrity baseline)
    print("\nCleaning...")
    for table_name, cfg in source_config["tables"].items():
        pk = cfg.get("pk")
        if pk and pk in frames[table_name].columns:
            before = len(frames[table_name])
            frames[table_name] = frames[table_name].dropna(subset=[pk])
            dropped = before - len(frames[table_name])
            if dropped:
                print(f"  {table_name}: dropped {dropped} rows with null PK '{pk}'")

    # 4. Audit nulls — document instead of blindly dropping
    # Remaining nulls are expected (e.g. undelivered orders have no delivery date)
    print("\nNull counts per table:")
    for name, df in frames.items():
        nulls = {col: int(n) for col, n in df.isnull().sum().items() if n > 0}
        if nulls:
            print(f"  {name}: {nulls}")
        else:
            print(f"  {name}: no nulls")

    # 5. Duplicate primary-key check — deduplicate if any found
    print("\nDuplicate key checks:")
    for table_name, cfg in source_config["tables"].items():
        pk = cfg.get("pk")
        if pk and pk in frames[table_name].columns:
            dupes = frames[table_name][pk].duplicated().sum()
            if dupes:
                print(f"  {table_name}.{pk}: {dupes} duplicates found — deduplicating (keeping first)")
                frames[table_name] = frames[table_name].drop_duplicates(subset=[pk], keep="first")
            else:
                print(f"  {table_name}.{pk}: clean")

    # 6. Write to SQLite
    print(f"\nWriting to SQLite -> {db_path}")
    conn = sqlite3.connect(db_path)
    for table_name, df in frames.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"  Wrote {table_name}: {len(df):,} rows")

    conn.commit()
    print(f"\nDone. Database written to {db_path}")
    return conn


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
        },
        "order_items": {
            "path": RAW_DIR / "olist_order_items_dataset.csv",
            "pk": None,  # composite key (order_id + order_item_id), checked via verify_db.py
            "date_cols": [],
        },
        "payments": {
            "path": RAW_DIR / "olist_order_payments_dataset.csv",
            "pk": None,
            "date_cols": [],
        },
        "reviews": {
            "path": RAW_DIR / "olist_order_reviews_dataset.csv",
            "pk": "review_id",
            "date_cols": ["review_creation_date", "review_answer_timestamp"],
        },
        "products": {
            "path": RAW_DIR / "olist_products_dataset.csv",
            "pk": "product_id",
            "date_cols": [],
        },
        "sellers": {
            "path": RAW_DIR / "olist_sellers_dataset.csv",
            "pk": "seller_id",
            "date_cols": [],
        },
        # Helper table — consumed by transform, not written to DB
        "category_translation": {
            "path": RAW_DIR / "product_category_name_translation.csv",
            "pk": None,
            "date_cols": [],
        },
    },
    "transforms": [_translate_product_categories],
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