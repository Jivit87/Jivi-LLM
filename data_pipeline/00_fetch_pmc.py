"""
Download evenly spaced Biomed-Enriched commercial shards one at a time (the connection is ~3.5 MB/s, so each 1.9 GB shard
takes ~9 min). Each finished shard is processed by `01_pmc_select.py` in a separate subprocess while the next one downloads.

Usage: python data_pipeline/00_fetch_pmc.py --shards 0 4 8 12 16 20 24
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import INTERIM, RAW, fetch  # noqa: E402

REPO, N = "almanach/Biomed-Enriched", 26


def expected_sizes():
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(REPO, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--tau-edu", type=float, default=4.15, help="candidate cutoff; final cut is made later")
    a = ap.parse_args()
    sizes = expected_sizes()
    running = []

    def reap(limit):
        while len(running) > limit:
            i, p = running.pop(0)
            rc = p.wait()
            print(f"shard {i}: processing finished rc={rc}", flush=True)

    for i in a.shards:
        out = os.path.join(INTERIM, "pmc_oa", f"commercial-{i:05d}.jsonl")
        if os.path.exists(out):
            print(f"shard {i}: already processed", flush=True)
            continue
        rel = f"data/commercial-{i:05d}-of-{N:05d}.parquet"
        dest = os.path.join(RAW, "pmc_oa", "_shards", rel)
        if not (os.path.exists(dest) and os.path.getsize(dest) == sizes[rel]):
            print(f"shard {i}: downloading", flush=True)
            fetch(f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}", dest)
        print(f"shard {i}: downloaded, processing in background", flush=True)
        reap(a.max_parallel - 1)
        running.append((i, subprocess.Popen([sys.executable, os.path.join(HERE, "01_pmc_select.py"), "--shards", str(i),
                                             "--delete-shards", "--tau-edu", str(a.tau_edu)])))
    reap(0)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
