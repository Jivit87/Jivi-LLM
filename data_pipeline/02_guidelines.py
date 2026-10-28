"""
Extract license-cleared clinical guidelines from the Meditron `epfl-llm/guidelines` open release.

Included by default: CDC (public domain, mostly) and WikiDoc (CC BY-SA 3.0).
Excluded: WHO (CC BY-NC-SA), NICE (licence excludes AI use), CCO (no reproduction), `pubmed` (overlaps PMC pipeline),
CMA / ICRC / SPOR (license not yet checked). See data/medical/manifest/licenses.csv.

Usage: python data_pipeline/02_guidelines.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, RAW, est_tokens, normalize_ws, read_jsonl, save_report, write_jsonl  # noqa: E402

INCLUDE = {"cdc", "wikidoc"}
LICENSE = {"cdc": "US-gov public domain (mostly)", "wikidoc": "CC BY-SA 3.0"}
MIN_CHARS = {"cdc": 1500, "wikidoc": 1500}  # drops WikiDoc stubs


def main():
    src = os.path.join(RAW, "guidelines", "open_guidelines.jsonl")
    counts = {}

    def docs():
        for d in read_jsonl(src):
            s = d["source"]
            c = counts.setdefault(s, {"seen": 0, "kept": 0, "chars": 0})
            c["seen"] += 1
            if s not in INCLUDE:
                continue
            text = normalize_ws(d.get("clean_text") or "")
            if len(text) < MIN_CHARS[s]:
                continue
            title = (d.get("title") or "").strip()
            if title:
                text = f"# {title}\n\n{text}"
            c["kept"] += 1
            c["chars"] += len(text)
            yield {"id": f"guideline:{d['id']}", "source": f"guidelines_{s}", "license": LICENSE[s],
                   "title": title, "text": text}

    out = os.path.join(INTERIM, "guidelines", "guidelines.jsonl")
    n = write_jsonl(out, docs())
    for c in counts.values():
        c["est_tokens"] = est_tokens(c["chars"])
    print(json.dumps(counts, indent=1))
    print(f"wrote {n} docs, ~{sum(c['est_tokens'] for c in counts.values())/1e6:.1f}M tokens -> {out}")
    save_report("guidelines_extract.json", counts)


if __name__ == "__main__":
    main()
