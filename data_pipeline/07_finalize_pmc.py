"""
Final PMC selection: rank all candidate articles from the processed shards and keep the best ones within a token budget.

Per-shard candidate files come from 01_pmc_select.py (generous cutoff). Here we:
  1. drop duplicate article ids and exact-duplicate texts
  2. score each article: educational mean, plus a small bonus for clinical cases and reviews
  3. fill quotas (clinical cases <= --case-share, reviews <= --review-share), then the rest by score
Output: data/medical/filtered/pmc_oa/pmc_selected.jsonl + a report.

Usage: python data_pipeline/07_finalize_pmc.py --target-tokens 280e6
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CHARS_PER_TOKEN, FILTERED, INTERIM, save_report  # noqa: E402

BONUS = {"clinical_case": 0.25, "review": 0.10, "study": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=float, default=280e6)
    ap.add_argument("--case-share", type=float, default=0.10, help="max share of chars from clinical-case articles")
    ap.add_argument("--review-share", type=float, default=0.30, help="max share of chars from review articles")
    ap.add_argument("--min-case-edu", type=float, default=3.9, help="clinical cases below this are never taken")
    ap.add_argument("--min-case-share", type=float, default=0.06, help="reserve this share for the best clinical cases")
    ap.add_argument("--min-review-share", type=float, default=0.15, help="reserve this share for the best reviews")
    a = ap.parse_args()
    budget = a.target_tokens * CHARS_PER_TOKEN

    files = sorted(glob.glob(os.path.join(INTERIM, "pmc_oa", "commercial-*.jsonl")))
    print(f"{len(files)} candidate files", flush=True)
    meta, seen_ids, seen_hash, dups = [], set(), set(), 0
    for fi, path in enumerate(files):
        with open(path, encoding="utf-8") as f:
            for li, line in enumerate(f):
                d = json.loads(line)
                h = hashlib.md5(d["text"].encode()).hexdigest()
                if d["id"] in seen_ids or h in seen_hash:
                    dups += 1
                    continue
                seen_ids.add(d["id"])
                seen_hash.add(h)
                score = d["edu_mean"] + BONUS[d["doc_type"]]
                meta.append((score, len(d["text"]), d["doc_type"], fi, li))
    print(f"{len(meta)} unique candidates ({dups} duplicates dropped)", flush=True)

    meta.sort(key=lambda x: -x[0])
    chosen, used, by_type = set(), 0, Counter()
    cap = {"clinical_case": a.case_share * budget, "review": a.review_share * budget, "study": budget}
    reserve = {"clinical_case": a.min_case_share * budget, "review": a.min_review_share * budget}

    def take(score, chars, ty, fi, li):
        nonlocal used
        chosen.add((fi, li))
        by_type[ty] += chars
        used += chars

    # phase 1: reserved shares for clinical cases and reviews (best-scoring first)
    for score, chars, ty, fi, li in meta:
        if ty in reserve and by_type[ty] < reserve[ty]:
            if ty == "clinical_case" and score - BONUS[ty] < a.min_case_edu:
                continue
            take(score, chars, ty, fi, li)
    # phase 2: everything else by global score, respecting caps
    for score, chars, ty, fi, li in meta:
        if used >= budget:
            break
        if (fi, li) in chosen:
            continue
        if ty == "clinical_case" and score - BONUS[ty] < a.min_case_edu:
            continue
        if by_type[ty] + chars > cap[ty]:
            continue
        take(score, chars, ty, fi, li)
    cutoff = min(m[0] for m in meta if (m[3], m[4]) in chosen and m[2] == "study")

    out = os.path.join(FILTERED, "pmc_oa", "pmc_selected.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    per_shard = Counter()
    with open(out + ".tmp", "w", encoding="utf-8") as w:
        for fi, path in enumerate(files):
            with open(path, encoding="utf-8") as f:
                for li, line in enumerate(f):
                    if (fi, li) in chosen:
                        w.write(line)
                        per_shard[os.path.basename(path)] += 1
    os.replace(out + ".tmp", out)
    rep = {"files": len(files), "candidates": len(meta), "selected": len(chosen),
           "est_tokens": int(used / CHARS_PER_TOKEN), "score_cutoff": round(cutoff, 3),
           "tokens_by_type": {k: int(v / CHARS_PER_TOKEN) for k, v in by_type.items()},
           "docs_per_shard": dict(per_shard), "params": vars(a)}
    print(json.dumps(rep, indent=1))
    save_report("pmc_final_selection.json", rep)


if __name__ == "__main__":
    main()
