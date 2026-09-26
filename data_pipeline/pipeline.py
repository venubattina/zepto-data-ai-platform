import sqlite3
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

FIXED_GBP_TO_INR = 105.50
CATEGORIES = [
    ("Travel", "http://books.toscrape.com/catalogue/category/books/travel_2/index.html"),
    ("Mystery", "http://books.toscrape.com/catalogue/category/books/mystery_3/index.html"),
    ("Historical Fiction", "http://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html"),
    ("Sequential Art", "http://books.toscrape.com/catalogue/category/books/sequential-art_5/index.html"),
]

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def scrape_catalog():
    records = []
    for cat_name, url in CATEGORIES:
        curr_url = url
        while curr_url:
            resp = requests.get(curr_url, timeout=10)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.content, "html.parser")
            articles = soup.find_all("article", class_="product_pod")
            for art in articles:
                title = art.h3.a["title"]
                price_text = art.find("p", class_="price_color").get_text(strip=True)
                rating_class = [c for c in art.find("p", class_="star-rating")["class"] if c != "star-rating"][0]
                stock_text = art.find("p", class_="instock availability").get_text(strip=True)
                records.append({
                    "title": title,
                    "price_raw": price_text,
                    "rating_raw": rating_class,
                    "availability_raw": stock_text,
                    "category": cat_name
                })
            next_btn = soup.find("li", class_="next")
            if next_btn:
                next_page = next_btn.a["href"]
                curr_url = "/".join(curr_url.split("/")[:-1]) + "/" + next_page
            else:
                curr_url = None
    return pd.DataFrame(records)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    def parse_price(val):
        cleaned = re.sub(r"[^\d.]", "", str(val))
        return float(cleaned) if cleaned else None

    df["price_gbp"] = df["price_raw"].apply(parse_price)
    if df["price_gbp"].isna().any():
        med_val = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(med_val)

    df["rating"] = df["rating_raw"].map(RATING_MAP).fillna(3).astype(int)
    df["in_stock"] = df["availability_raw"].str.contains("In stock", case=False).astype(int)
    df["price_inr"] = (df["price_gbp"] * FIXED_GBP_TO_INR).round(2)
    return df

def init_db(df: pd.DataFrame, db_path="data_pipeline/zepto_catalog.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories (category_id)
    );
    """)
    conn.commit()

    unique_categories = df["category"].drop_duplicates().tolist()
    for cat in unique_categories:
        cur.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat,))
    conn.commit()

    cat_map = pd.read_sql("SELECT category_name, category_id FROM categories", conn).set_index("category_name")["category_id"].to_dict()
    df["category_id"] = df["category"].map(cat_map)

    book_records = df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]].to_dict(orient="records")
    cur.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (:title, :price_gbp, :price_inr, :rating, :in_stock, :category_id)
    """, book_records)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Scraping books.toscrape.com...")
    raw_df = scrape_catalog()
    print(f"Scraped {len(raw_df)} books. Cleaning data...")
    cleaned_df = clean_data(raw_df)
    init_db(cleaned_df)
    print(f"Data pipeline complete. zepto_catalog.db created with {len(cleaned_df)} rows.")
