import sqlite3
import pandas as pd

conn = sqlite3.connect("data_pipeline/zepto_catalog.db")

queries = {
    "q1_select_where": "SELECT title, price_inr, rating FROM books WHERE rating >= 4 LIMIT 5;",
    "q2_order_limit": "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 5;",
    "q3_distinct": "SELECT DISTINCT rating FROM books ORDER BY rating ASC;",
    "q4_between": "SELECT title, price_inr FROM books WHERE price_inr BETWEEN 2000.00 AND 3000.00 LIMIT 5;",
    "q5_join": """
        SELECT b.title, b.price_inr, b.rating, c.category_name 
        FROM books b 
        INNER JOIN categories c ON b.category_id = c.category_id 
        WHERE b.rating = 5 
        ORDER BY b.price_inr DESC 
        LIMIT 5;
    """
}

for name, sql in queries.items():
    res = pd.read_sql(sql, conn)
    print(f"\n--- {name} ---\n", res)

# Read query 5 via SQL
sql_join_result = pd.read_sql(queries["q5_join"], conn)

# Reproduce join using in-memory DataFrames
books_df = pd.read_sql("SELECT * FROM books", conn)
cats_df = pd.read_sql("SELECT * FROM categories", conn)

merged = pd.merge(books_df, cats_df, on="category_id", how="inner")
pandas_join_result = (
    merged[merged["rating"] == 5][["title", "price_inr", "rating", "category_name"]]
    .sort_values(by="price_inr", ascending=False)
    .head(5)
    .reset_index(drop=True)
)

print("\n--- Verifying SQL vs. Pandas Merge Equivalence ---")
assert sql_join_result.equals(pandas_join_result), "Equivalence check failed!"
print("Equivalence Confirmed: SQL query output and pd.merge produce identical results.")
conn.close()
