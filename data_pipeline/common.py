"""Shared helpers for the medical data pipeline (see docs/med_domain.md)."""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MED = os.path.join(ROOT, "data", "medical")
RAW = os.path.join(MED, "raw")
INTERIM = os.path.join(MED, "interim")
FILTERED = os.path.join(MED, "filtered")
REPORTS = os.path.join(MED, "reports")

CHARS_PER_TOKEN = 4.2  # rough GPT-2-BPE proxy; exact counts come after the custom tokenizer is trained


def est_tokens(n_chars):
    return int(n_chars / CHARS_PER_TOKEN)


def write_jsonl(path, docs):
    """Write docs atomically (tmp file then rename) so a crash never leaves a half-written shard."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def save_report(name, obj):
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, name), "w") as f:
        json.dump(obj, f, indent=2)


_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def normalize_ws(text):
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = _WS.sub(" ", text)
    return _NL.sub("\n\n", text).strip()


def fetch(url, dest, retries=20):
    """Resumable download with curl (HF's xet transfer stalled on this connection). Returns dest."""
    import subprocess
    import time

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sSL", "-C", "-", "--fail", "--connect-timeout", "30", "--speed-limit", "20000",
                            "--speed-time", "60", "-o", dest, url])
        if r.returncode == 0:
            return dest
        print(f"curl rc={r.returncode} on {os.path.basename(dest)}, retry {attempt + 1}", flush=True)
        time.sleep(5)
    raise RuntimeError(f"download failed: {url}")


import hashlib as _hashlib


def is_val(doc_id, pct=2):
    """Deterministic document-level validation split: about `pct` percent of documents, stable across runs."""
    return int(_hashlib.md5(doc_id.encode()).hexdigest(), 16) % 100 < pct


# where cleaned data lives after 08_dedupe_decontam.py
SOURCE_DIRS = {
    "pmc_oa": "pmc_oa_clean",
    "pubmed_abstracts": "pubmed_abstracts",
    "guidelines": "guidelines",
    "reference": "reference",
    "general_replay": "general_replay",
}


def iter_source(source):
    """Yield every cleaned document of a source."""
    import glob

    for path in sorted(glob.glob(os.path.join(FILTERED, SOURCE_DIRS[source], "*.jsonl"))):
        yield from read_jsonl(path)
