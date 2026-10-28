"""
Extract clean English abstracts from PubMed baseline files (https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/).

PubMed's terms say "some abstracts may be protected by copyright". To stay conservative we keep an abstract only if it has
no CopyrightInformation, or that line mentions a Creative Commons licence. We also keep recent, substantial, research-type
abstracts and skip retracted items, editorials, letters, comments, news, etc.

Usage: python data_pipeline/03_pubmed_abstracts.py --files 1000 1014 1028 ...
"""

import argparse
import gzip
import os
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, RAW, est_tokens, fetch, normalize_ws, save_report, write_jsonl  # noqa: E402

BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
YEAR = "26"  # baseline vintage in the file names, e.g. pubmed26n1000.xml.gz
BAD_TYPES = {"Retracted Publication", "Retraction of Publication", "Published Erratum", "Comment", "Editorial", "Letter",
             "News", "Newspaper Article", "Interview", "Biography", "Portrait", "Congress", "Lecture", "Retraction Notice",
             "Expression of Concern", "Duplicate Publication", "Legal Case"}
CC = re.compile(r"creative\s*commons|CC[\s-]BY|CC0", re.I)


def text_of(el):
    return "".join(el.itertext()).strip() if el is not None else ""


def parse(path, a, stats):
    with gzip.open(path, "rb") as f:
        for _, art in ET.iterparse(f, events=("end",)):
            if art.tag != "PubmedArticle":
                continue
            stats["articles"] += 1
            mc = art.find("MedlineCitation")
            pmid = text_of(mc.find("PMID"))
            ar = mc.find("Article")
            ab = ar.find("Abstract") if ar is not None else None
            reason = None
            if ab is None:
                reason = "no_abstract"
            else:
                langs = [text_of(x) for x in ar.findall("Language")]
                ptypes = {text_of(x) for x in ar.findall("PublicationTypeList/PublicationType")}
                year = text_of(ar.find("Journal/JournalIssue/PubDate/Year")) or text_of(mc.find("DateCompleted/Year"))
                crt = text_of(ab.find("CopyrightInformation"))
                if "eng" not in langs:
                    reason = "not_english"
                elif ptypes & BAD_TYPES:
                    reason = "bad_type"
                elif crt and not a.allow_copyrighted and not CC.search(crt):
                    reason = "copyright_notice"
                elif year and year.isdigit() and int(year) < a.min_year:
                    reason = "too_old"
            if reason is None:
                parts = []
                for t in ab.findall("AbstractText"):
                    label = (t.get("Label") or "").strip()
                    body = text_of(t)
                    if body:
                        parts.append(f"{label.capitalize()}: {body}" if label else body)
                abstract = normalize_ws("\n".join(parts))
                words = len(abstract.split())
                title = text_of(ar.find("ArticleTitle")).rstrip()
                has_mesh = mc.find("MeshHeadingList") is not None
                if words < a.min_words:
                    reason = "too_short"
                elif words > a.max_words:
                    reason = "too_long"
                elif a.require_mesh and not has_mesh:
                    reason = "no_mesh"
            if reason:
                stats["dropped"][reason] = stats["dropped"].get(reason, 0) + 1
                art.clear()
                continue
            stats["kept"] += 1
            stats["chars"] += len(abstract) + len(title)
            yield {"id": f"pmid:{pmid}", "source": "pubmed_abstracts", "license": "NLM terms; no publisher copyright notice",
                   "year": int(year) if year.isdigit() else None, "title": title,
                   "text": f"# {title}\n\n{abstract}"}
            art.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, nargs="+", required=True, help="baseline file numbers, e.g. 1000 1014")
    ap.add_argument("--min-year", type=int, default=2012)
    ap.add_argument("--min-words", type=int, default=150)
    ap.add_argument("--max-words", type=int, default=450)
    ap.add_argument("--require-mesh", action="store_true", help="only MEDLINE-indexed (curated) records")
    ap.add_argument("--allow-copyrighted", action="store_true", help="also keep abstracts with a publisher (c) line (grey area)")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    total = 0
    for n in a.files:
        name = f"pubmed{YEAR}n{n:04d}.xml.gz"
        out = os.path.join(INTERIM, "pubmed_abstracts", f"pubmed{YEAR}n{n:04d}.jsonl")
        if os.path.exists(out):
            print(f"{name}: already done", flush=True)
            continue
        raw = fetch(f"{BASE}/{name}", os.path.join(RAW, "pubmed_abstracts", name))
        stats = {"file": name, "articles": 0, "kept": 0, "chars": 0, "dropped": {}}
        write_jsonl(out, parse(raw, a, stats))
        stats["est_tokens"] = est_tokens(stats["chars"])
        total += stats["est_tokens"]
        print(f"{name}: kept {stats['kept']}/{stats['articles']} ~{stats['est_tokens']/1e6:.1f}M tok  drops={stats['dropped']}",
              flush=True)
        save_report(f"pubmed_{n:04d}.json", {**stats, "params": vars(a)})
        if not a.keep_raw:
            os.remove(raw)
    print(f"total ~{total/1e6:.1f}M tokens this run", flush=True)


if __name__ == "__main__":
    main()
