"""
Downloads several public domain books from Project Gutenberg
and combines them into one large training file.

Run with:
    .venv/bin/python download_data.py
"""

import os
import requests

# public domain books from Project Gutenberg (plain text URLs)
BOOKS = [
    ("the-verdict.txt",           "https://www.gutenberg.org/files/2500/2500-0.txt"),   # The Verdict — Edith Wharton
    ("pride-and-prejudice.txt",   "https://www.gutenberg.org/files/1342/1342-0.txt"),   # Pride and Prejudice — Austen
    ("adventures-huckleberry.txt","https://www.gutenberg.org/files/76/76-0.txt"),        # Huckleberry Finn — Twain
    ("sherlock-holmes.txt",       "https://www.gutenberg.org/files/1661/1661-0.txt"),    # Sherlock Holmes — Doyle
    ("frankenstein.txt",          "https://www.gutenberg.org/files/84/84-0.txt"),        # Frankenstein — Shelley
    ("great-expectations.txt",    "https://www.gutenberg.org/files/1400/1400-0.txt"),    # Great Expectations — Dickens
]

RAW_DIR = "data/raw"
COMBINED = "data/raw/combined.txt"

os.makedirs(RAW_DIR, exist_ok=True)

combined_text = ""

for filename, url in BOOKS:
    path = os.path.join(RAW_DIR, filename)

    if os.path.exists(path):
        print(f"Already downloaded: {filename}")
    else:
        print(f"Downloading {filename}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  Saved ({len(response.text):,} bytes)")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        combined_text += f.read() + "\n\n"

with open(COMBINED, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"\nCombined file: {COMBINED}")
print(f"Total size: {len(combined_text):,} bytes (~{len(combined_text)//4:,} tokens)")
