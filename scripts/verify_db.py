"""
Phase 1 - DB Verification Script
Quick sanity checks on analytics.db after load_data.py has run.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("db/analytics.db")

conn = sqlite3.connect(DB_PATH)

print("=== Tables in analytics.db ===")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print(tables.to_string(index=False))
print()

print("=== Row counts & columns ===")
for t in tables["name"]:
    count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM [{t}]", conn).iloc[0, 0]
    cols = pd.read_sql(f"PRAGMA table_info([{t}])", conn)["name"].tolist()
    print(f"  {t}: {count:,} rows")
    print(f"    cols: {cols}")
print()

print("=== Test Query 1: Order status breakdown with avg price ===")
q1 = pd.read_sql("""
    SELECT o.order_status,
           COUNT(*) as order_count,
           ROUND(AVG(oi.price), 2) as avg_item_price
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY o.order_status
    ORDER BY order_count DESC
""", conn)
print(q1.to_string(index=False))
print()

print("=== Test Query 2: Top 5 product categories by revenue ===")
q2 = pd.read_sql("""
    SELECT p.product_category_name_english as category,
           COUNT(oi.order_id) as items_sold,
           ROUND(SUM(oi.price), 2) as total_revenue
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    WHERE p.product_category_name_english IS NOT NULL
    GROUP BY category
    ORDER BY total_revenue DESC
    LIMIT 5
""", conn)
print(q2.to_string(index=False))
print()

print("=== Test Query 3: Orders with customer city ===")
q3 = pd.read_sql("""
    SELECT o.order_id,
           o.order_status,
           c.customer_city,
           c.customer_state,
           o.order_purchase_timestamp
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    LIMIT 5
""", conn)
print(q3.to_string(index=False))
print()

print("=== Test Query 4: Avg review score per seller ===")
q4 = pd.read_sql("""
    SELECT s.seller_id,
           s.seller_city,
           ROUND(AVG(r.review_score), 2) as avg_score,
           COUNT(r.review_id) as review_count
    FROM sellers s
    JOIN order_items oi ON s.seller_id = oi.seller_id
    JOIN reviews r ON oi.order_id = r.order_id
    GROUP BY s.seller_id, s.seller_city
    ORDER BY avg_score DESC
    LIMIT 5
""", conn)
print(q4.to_string(index=False))
print()

print("=== Test Query 5: Payment method breakdown ===")
q5 = pd.read_sql("""
    SELECT payment_type,
           COUNT(*) as transactions,
           ROUND(SUM(payment_value), 2) as total_value,
           ROUND(AVG(payment_value), 2) as avg_value
    FROM payments
    GROUP BY payment_type
    ORDER BY transactions DESC
""", conn)
print(q5.to_string(index=False))
print()

conn.close()
print("All verification queries passed.")
