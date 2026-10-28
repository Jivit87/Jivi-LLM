"""
Final cleaning pass over every source: exact dedupe, abstract-vs-PMC overlap removal, and benchmark decontamination.

All three use 13-word shingles. To keep it fast we only keep shingles whose hash is 0 mod 4 ("content-defined sampling"),
so any copied span of >= ~17 words is still caught with high probability and both sides always sample the same shingles.

  - decontamination: drop any document sharing a shingle with a benchmark test/validation/dev question or context
    (PubMedQA, MedMCQA, MedQA, MMLU-medical). Train splits are not used for this check.
  - abstract/PMC overlap: drop a PubMed abstract if >= 50% of its shingles occur in the selected PMC text.
  - exact dedupe: normalized text hash, across all sources.

IMPORTANT: run with PYTHONHASHSEED=0 so hashes are stable:   PYTHONHASHSEED=0 python data_pipeline/08_dedupe_decontam.py
Output: data/medical/filtered/<source>/*.jsonl and data/medical/reports/clean_report.json
"""

import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FILTERED, INTERIM, MED, est_tokens, save_report  # noqa: E402

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("run with PYTHONHASHSEED=0 (stable hashing across processes)")

N = 13
WORD = re.compile(r"[a-z0-9]+")
P = np.uint64(1000003)
SAMPLE_MASK = np.uint64(3)


def shingles(text):
    """Sampled 13-gram hashes (uint64 array) of a text."""
    words = WORD.findall(text.lower())
    if len(words) < N:
        return np.empty(0, dtype=np.uint64)
    h = np.fromiter((hash(w) & 0xFFFFFFFFFFFFFFFF for w in words), dtype=np.uint64, count=len(words))
    m = len(h) - N + 1
    acc = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(N):
            acc = acc * P + h[j:j + m]
        acc ^= acc >> np.uint64(29)
        acc *= np.uint64(0x9E3779B97F4A7C15)
        acc ^= acc >> np.uint64(32)
    return acc[(acc & SAMPLE_MASK) == 0]


def member(sorted_arr, x):
    """Boolean mask: which elements of x occur in sorted_arr (no re-sorting of the big array)."""
    if len(x) == 0 or len(sorted_arr) == 0:
        return np.zeros(len(x), dtype=bool)
    idx = np.searchsorted(sorted_arr, x)
    idx[idx == len(sorted_arr)] = len(sorted_arr) - 1
    return sorted_arr[idx] == x


def eval_texts():
    """Question/context text of benchmark splits that must not appear in training data."""
    out = []
    for name, splits in [("pubmedqa", ["train"]), ("medmcqa", ["test", "validation"]), ("medqa", ["test"]),
                         ("mmlu_medical", ["test", "validation", "dev"])]:
        for sp in splits:
            path = os.path.join(MED, "eval", name, f"{sp}.jsonl")
            if not os.path.exists(path):
                print(f"WARNING: missing eval set {path}", flush=True)
                continue
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                parts = []
                if name == "pubmedqa":
                    ctx = r.get("context") or {}
                    parts += [r.get("question", "")] + list(ctx.get("contexts", []))
                elif name == "medmcqa":
                    parts += [r.get("question", ""), r.get("exp") or ""]
                elif name == "medqa":
                    parts += [r.get("question", "")]
                else:
                    parts += [r.get("question", "")]
                out.append(" ".join(p for p in parts if p))
    return out


_EVAL = None


def _init(eval_arr):
    global _EVAL
    _EVAL = eval_arr


def work(line):
    d = json.loads(line)
    sh = shingles(d["text"])
    contaminated = bool(member(_EVAL, sh).any())
    h = hashlib.md5(" ".join(WORD.findall(d["text"].lower())).encode()).hexdigest()
    return line, sh, contaminated, h, len(d["text"])


def process(path_glob, source, eval_arr, seen_hash, pmc_set=None, collect=False):
    files = sorted(glob.glob(path_glob))
    rep = {"in": 0, "contaminated": 0, "exact_dup": 0, "pmc_overlap": 0, "kept": 0, "chars": 0}
    kept_shingles = []
    outdir = os.path.join(FILTERED, source)
    os.makedirs(outdir, exist_ok=True)
    with mp.Pool(min(20, os.cpu_count()), initializer=_init, initargs=(eval_arr,)) as pool:
        for path in files:
            out = os.path.join(outdir, os.path.basename(path))
            with open(path, encoding="utf-8") as f, open(out + ".tmp", "w", encoding="utf-8") as w:
                for line, sh, contam, h, nchars in pool.imap(work, f, chunksize=64):
                    rep["in"] += 1
                    if contam:
                        rep["contaminated"] += 1
                    elif h in seen_hash:
                        rep["exact_dup"] += 1
                    elif pmc_set is not None and len(sh) and member(pmc_set, sh).mean() >= 0.5:
                        rep["pmc_overlap"] += 1
                    else:
                        seen_hash.add(h)
                        w.write(line if line.endswith("\n") else line + "\n")
                        rep["kept"] += 1
                        rep["chars"] += nchars
                        if collect:
                            kept_shingles.append(sh)
            os.replace(out + ".tmp", out)
    rep["est_tokens"] = est_tokens(rep["chars"])
    print(f"{source}: {rep}", flush=True)
    return rep, kept_shingles


def main():
    ev = eval_texts()
    print(f"{len(ev)} benchmark items", flush=True)
    eval_arr = np.unique(np.concatenate([shingles(t) for t in ev]))
    print(f"{len(eval_arr)} benchmark shingles", flush=True)

    seen, report = set(), {}
    report["pmc_oa"], pmc_sh = process(os.path.join(FILTERED, "pmc_oa", "pmc_selected.jsonl"), "pmc_oa_clean",
                                       eval_arr, seen, collect=True)
    pmc_set = np.unique(np.concatenate(pmc_sh)) if pmc_sh else None
    del pmc_sh
    print(f"PMC shingle set: {0 if pmc_set is None else len(pmc_set)}", flush=True)
    report["pubmed_abstracts"], _ = process(os.path.join(INTERIM, "pubmed_abstracts", "*.jsonl"), "pubmed_abstracts",
                                            eval_arr, seen, pmc_set=pmc_set)
    report["guidelines"], _ = process(os.path.join(INTERIM, "guidelines", "*.jsonl"), "guidelines", eval_arr, seen)
    report["reference"], _ = process(os.path.join(INTERIM, "reference", "*.jsonl"), "reference", eval_arr, seen)
    report["general_replay"], _ = process(os.path.join(INTERIM, "general_replay", "*.jsonl"), "general_replay",
                                          eval_arr, seen)
    save_report("clean_report.json", report)
    print(json.dumps({k: v["est_tokens"] for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
