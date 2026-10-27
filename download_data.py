"""
Downloads public domain books from Project Gutenberg, strips the licence
header/footer, and combines them into one training file.

Run with:
    .venv/bin/python download_data.py
"""

import os
import re
import time
import requests

# (filename, Gutenberg ebook id)
BOOKS = [
    ("the-verdict", 2500), ("pride-and-prejudice", 1342), ("huckleberry-finn", 76),
    ("sherlock-holmes", 1661), ("frankenstein", 84), ("great-expectations", 1400),
    ("emma", 158), ("sense-and-sensibility", 161), ("mansfield-park", 141),
    ("persuasion", 105), ("northanger-abbey", 121),
    ("tom-sawyer", 74), ("prince-and-pauper", 1837), ("connecticut-yankee", 86),
    ("life-on-the-mississippi", 245), ("innocents-abroad", 3176),
    ("oliver-twist", 730), ("david-copperfield", 766), ("tale-of-two-cities", 98),
    ("bleak-house", 1023), ("little-dorrit", 963), ("nicholas-nickleby", 967),
    ("our-mutual-friend", 883), ("pickwick-papers", 580), ("christmas-carol", 46),
    ("dombey-and-son", 821), ("old-curiosity-shop", 700),
    ("jane-eyre", 1260), ("wuthering-heights", 768), ("villette", 9182),
    ("tenant-of-wildfell-hall", 969), ("agnes-grey", 767),
    ("middlemarch", 145), ("mill-on-the-floss", 6688), ("silas-marner", 550),
    ("vanity-fair", 599), ("tess", 110), ("jude-the-obscure", 153),
    ("far-from-the-madding-crowd", 107), ("return-of-the-native", 122),
    ("moby-dick", 2701), ("scarlet-letter", 33), ("house-seven-gables", 77),
    ("dracula", 345), ("dr-jekyll-mr-hyde", 43), ("treasure-island", 120),
    ("kidnapped", 421), ("island-of-doctor-moreau", 159), ("time-machine", 35),
    ("war-of-the-worlds", 36), ("invisible-man", 5230),
    ("hound-of-baskervilles", 2852), ("adventures-sherlock", 1661),
    ("memoirs-sherlock", 834), ("return-sherlock", 108), ("study-in-scarlet", 244),
    ("sign-of-four", 2097), ("lost-world", 139),
    ("alice-wonderland", 11), ("through-looking-glass", 12), ("peter-pan", 16),
    ("wizard-of-oz", 55), ("secret-garden", 113), ("little-women", 514),
    ("anne-of-green-gables", 45), ("jungle-book", 236), ("black-beauty", 271),
    ("wind-in-the-willows", 289), ("heidi", 1448),
    ("three-musketeers", 1257), ("count-of-monte-cristo", 1184),
    ("les-miserables", 135), ("hunchback-notre-dame", 2610),
    ("war-and-peace", 2600), ("anna-karenina", 1399),
    ("crime-and-punishment", 2554), ("brothers-karamazov", 28054),
    ("idiot", 2638), ("don-quixote", 996), ("ulysses", 4300),
    ("portrait-of-the-artist", 4217), ("dubliners", 2814),
    ("heart-of-darkness", 219), ("lord-jim", 5658), ("secret-agent", 974),
    ("picture-of-dorian-gray", 174), ("importance-of-being-earnest", 844),
    ("call-of-the-wild", 215), ("white-fang", 910), ("sea-wolf", 1074),
    ("last-of-the-mohicans", 940), ("deerslayer", 3285),
    ("gullivers-travels", 829), ("robinson-crusoe", 521),
    ("ivanhoe", 82), ("rob-roy", 7025), ("kenilworth", 4895),
    ("age-of-innocence", 541), ("house-of-mirth", 284), ("ethan-frome", 4517),
    ("my-antonia", 242), ("o-pioneers", 24), ("sister-carrie", 233),
    ("main-street", 543), ("babbitt", 1156), ("the-awakening", 160),
    ("walden", 205), ("leaves-of-grass", 1322),
    ("origin-of-species", 1228), ("autobiography-franklin", 148),
    ("the-prince", 1232), ("republic-plato", 1497), ("meditations", 2680),
    ("paradise-lost", 20), ("odyssey", 3160), ("iliad", 6130),
]

RAW_DIR = "data/raw"
COMBINED = "data/raw/combined.txt"

START_RE = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)


def clean(text):
    text = text.replace("\r\n", "\n").lstrip("﻿")
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[:m.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch(book_id):
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                r.encoding = "utf-8"
                return r.text
        except requests.RequestException:
            pass
    return None


os.makedirs(RAW_DIR, exist_ok=True)

parts, seen_ids = [], set()
for name, book_id in BOOKS:
    if book_id in seen_ids:  # skip duplicate ids in the list
        continue
    seen_ids.add(book_id)
    path = os.path.join(RAW_DIR, f"{name}.txt")

    if os.path.exists(path):
        print(f"Already downloaded: {name}")
    else:
        print(f"Downloading {name} ({book_id})...")
        text = fetch(book_id)
        if text is None:
            print("  FAILED, skipping")
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        time.sleep(0.5)  # be polite to Gutenberg

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        parts.append(clean(f.read()))

combined_text = "\n\n".join(parts) + "\n"
with open(COMBINED, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"\nBooks combined: {len(parts)}")
print(f"Combined file: {COMBINED}")
print(f"Total size: {len(combined_text):,} bytes (~{len(combined_text)//4:,} tokens)")
