"""
General-English "replay" text from FineWeb-Edu (sample-10BT, ODC-By). Streams only as many documents as needed.

Gives the from-scratch model fluent everyday English and guards against the corpus being pure research prose.
We take documents with a high educational score and drop ones that are already medical-looking-heavy is NOT done here;
this slice is meant to be general.

Usage: python data_pipeline/04_general_replay.py --tokens 75e6
"""

import argparse
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, est_tokens, normalize_ws, save_report, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=float, default=75e6)
    ap.add_argument("--min-score", type=float, default=3.2)
    ap.add_argument("--min-chars", type=int, default=1500)
    ap.add_argument("--shard-docs", type=int, default=50_000)
    a = ap.parse_args()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    stats = {"seen": 0, "kept": 0, "tokens": 0}
    shard_i, buf = 0, []

    def flush():
        nonlocal shard_i, buf
        if buf:
            write_jsonl(os.path.join(INTERIM, "general_replay", f"fineweb_edu_{shard_i:03d}.jsonl"), buf)
            shard_i, buf = shard_i + 1, []

    for ex in ds:
        stats["seen"] += 1
        if ex["score"] < a.min_score or len(ex["text"]) < a.min_chars:
            continue
        text = normalize_ws(ex["text"])
        buf.append({"id": f"fwedu:{ex['id']}", "source": "general_replay", "license": "ODC-By (FineWeb-Edu)",
                    "url": ex.get("url"), "edu_score": ex["score"], "text": text})
        stats["kept"] += 1
        stats["tokens"] += ex.get("token_count") or est_tokens(len(text))
        if len(buf) >= a.shard_docs:
            flush()
            print(f"kept {stats['kept']} docs, {stats['tokens']/1e6:.1f}M tokens (seen {stats['seen']})", flush=True)
        if stats["tokens"] >= a.tokens:
            break
    flush()
    print(stats)
    save_report("general_replay.json", {**stats, "params": vars(a)})


if __name__ == "__main__":
    main()
