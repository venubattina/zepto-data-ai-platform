# Module 1 — Data Pipeline (/data_pipeline)

## Overview
Scrapes product catalog data from books.toscrape.com across 4 categories, parses and cleans attributes, converts currency using a project-defined fixed rate, and persists the data into a normalized SQLite relational schema.

## Currency Conversion
- **Fixed Baseline Rate**: 1 GBP = 105.50 INR (project constant).
- Handled locally in `clean_data()`: `df["price_inr"] = (df["price_gbp"] * 105.50).round(2)`.

## Relational Schema Design
Implemented in SQLite with foreign keys enforced:
- `categories (category_id PK, category_name UNIQUE)`
- `books (book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)`

## Query Benchmarks & Equivalence Proof
Executed in `query_benchmark.py`:
1. `SELECT / WHERE`: Filters books by rating >= 4.
2. `ORDER BY / LIMIT`: Top 5 highest-priced items in INR.
3. `DISTINCT`: Distinct rating values (1 through 5).
4. `BETWEEN`: Items priced between 2000.00 and 3000.00 INR.
5. `JOIN`: Joins `books` and `categories` on `category_id`.
- Equivalence verified between `pd.read_sql` and `pd.merge` using `assert sql_join_result.equals(pandas_join_result)`.
