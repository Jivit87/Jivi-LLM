"""
Select high-quality full articles from the Biomed-Enriched *commercial* split (PMC OA, CC BY / CC BY-SA / CC0 family).

Paragraphs arrive contiguous and ordered by article, so we rebuild each article, then keep it only if its average
educational score clears a threshold (lower thresholds for reviews and clinical cases, which the dataset's authors found
helpful). Output is one jsonl per input shard under data/medical/interim/pmc_oa/.

Usage:
    python data_pipeline/01_pmc_select.py --shards 0 13 25 --tau-edu 4.2          # calibration subset
    python data_pipeline/01_pmc_select.py --all --tau-edu 4.2 --delete-shards     # full run
"""

import argparse
import os
import re
import sys
import time

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, RAW, est_tokens, fetch, normalize_ws, save_report, write_jsonl  # noqa: E402

REPO = "almanach/Biomed-Enriched"
N_SHARDS = 26
COLS = ["article_id", "text", "language", "section_title", "document_type", "educational_score", "license_url"]

# back-matter sections that carry no medical knowledge
DROP_SECTION = re.compile(
    r"^\W*(\d+(\.\d+)*\.?\s*)?(references?|bibliography|acknowledg\w*|funding|financial support|conflicts? of interests?|"
    r"competing interests?|declaration\w*|author(s)?'? contributions?|ethic\w*|data availability|availability of data.*|"
    r"supplementary.*|abbreviations?|disclosure\w*|consent.*|institutional review.*|informed consent.*)\W*$",
    re.I,
)
NUMBERING = re.compile(r"^\s*\d+(\.\d+)*\.?\s+")


def shard_path(i):
    return os.path.join(RAW, "pmc_oa", "_shards", "data", f"commercial-{i:05d}-of-{N_SHARDS:05d}.parquet")


def download(i):
    dest = shard_path(i)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/commercial-{i:05d}-of-{N_SHARDS:05d}.parquet"
    return fetch(url, dest)


def build_article(rows, a):
    """rows: list of (text, section, doc_type, edu, license). Return a doc dict or None."""
    kept = [r for r in rows if r[3] >= a.p_min and not DROP_SECTION.match(r[1] or "")]
    if len(kept) < a.min_paras:
        return None
    n = len(kept)
    mean_edu = sum(r[3] for r in kept) / n
    frac_review = sum(r[2] == "review" for r in kept) / n
    frac_case = sum(r[2] == "clinical_case" for r in kept) / n
    tau = a.tau_edu
    if frac_case > 0.5:
        tau -= a.case_delta
    elif frac_review > 0.5:
        tau -= a.review_delta
    if mean_edu < tau:
        return None
    parts, last = [], None
    for text, sec, _, _, _ in kept:
        sec = NUMBERING.sub("", sec or "").strip()
        if sec and sec != last:
            parts.append(f"## {sec}")
            last = sec
        parts.append(text.strip())
    text = normalize_ws("\n\n".join(parts))
    if len(text) < a.min_chars:
        return None
    dtype = "clinical_case" if frac_case > 0.5 else "review" if frac_review > 0.5 else "study"
    return {"source": "pmc_oa", "license": kept[0][4], "n_paras": n, "edu_mean": round(mean_edu, 3),
            "doc_type": dtype, "text": text}


def process_shard(i, a):
    out = os.path.join(INTERIM, "pmc_oa", f"commercial-{i:05d}.jsonl")
    if os.path.exists(out) and not a.force:
        print(f"shard {i}: exists, skipping", flush=True)
        return None
    t0 = time.time()
    path = shard_path(i)
    if not os.path.exists(path):
        path = download(i)
    t_dl = time.time() - t0
    pf = pq.ParquetFile(path)
    stats = {"shard": i, "paras_seen": 0, "articles_seen": 0, "articles_kept": 0, "chars_kept": 0,
             "by_type": {}}

    def docs():
        cur_id, rows = None, []

        def flush():
            if cur_id is None or not rows:
                return None
            stats["articles_seen"] += 1
            d = build_article(rows, a)
            if d is None:
                return None
            d["id"] = f"pmc:{cur_id}"
            stats["articles_kept"] += 1
            stats["chars_kept"] += len(d["text"])
            stats["by_type"][d["doc_type"]] = stats["by_type"].get(d["doc_type"], 0) + 1
            return d

        for batch in pf.iter_batches(batch_size=50_000, columns=COLS):
            cols = {c: batch.column(c).to_pylist() for c in COLS}
            for k in range(batch.num_rows):
                stats["paras_seen"] += 1
                if cols["language"][k] != "en":
                    continue
                aid = cols["article_id"][k]
                if aid != cur_id:
                    d = flush()
                    if d:
                        yield d
                    cur_id, rows = aid, []
                rows.append((cols["text"][k], cols["section_title"][k], cols["document_type"][k],
                             cols["educational_score"][k], cols["license_url"][k]))
        d = flush()
        if d:
            yield d

    n = write_jsonl(out, docs())
    stats.update(docs_written=n, est_tokens=est_tokens(stats["chars_kept"]),
                 seconds=round(time.time() - t0), download_seconds=round(t_dl))
    print(f"shard {i}: kept {stats['articles_kept']}/{stats['articles_seen']} articles, "
          f"~{stats['est_tokens']/1e6:.1f}M tokens, {stats['seconds']}s", flush=True)
    save_report(f"pmc_select_shard{i:02d}.json", {**stats, "params": vars(a)})
    if a.delete_shards and os.path.exists(path):
        os.remove(path)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tau-edu", type=float, default=3.9, help="min mean educational score; keep this generous, final cut is made in 01b")
    ap.add_argument("--review-delta", type=float, default=0.2, help="threshold relief for review articles")
    ap.add_argument("--case-delta", type=float, default=0.4, help="threshold relief for clinical-case articles")
    ap.add_argument("--p-min", type=float, default=3.5, help="drop paragraphs scoring below this")
    ap.add_argument("--min-paras", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=4000)
    ap.add_argument("--delete-shards", action="store_true", help="delete each parquet after processing")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    shards = list(range(N_SHARDS)) if a.all else a.shards
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(process_shard, shards, [a] * len(shards)))
    else:
        for i in shards:
            process_shard(i, a)


if __name__ == "__main__":
    main()
