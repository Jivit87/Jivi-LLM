"""
Download medical benchmark sets into data/medical/eval/. They are used (a) to decontaminate the training corpus and
(b) later for evaluation. Never train on these test splits.

Usage: python data_pipeline/06_fetch_eval.py
"""

import json
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MED, save_report  # noqa: E402

MMLU_MEDICAL = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "medical_genetics",
                "professional_medicine", "nutrition", "virology"]


def dump(name, split_name, rows):
    d = os.path.join(MED, "eval", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{split_name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def main():
    rep = {}
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
        for sp in ds:
            rep[f"pubmedqa/{sp}"] = dump("pubmedqa", sp, [dict(x) for x in ds[sp]])
    except Exception as e:
        rep["pubmedqa"] = f"FAILED {type(e).__name__}: {str(e)[:150]}"
    try:
        ds = load_dataset("openlifescienceai/medmcqa")
        for sp in ds:
            rep[f"medmcqa/{sp}"] = dump("medmcqa", sp, [dict(x) for x in ds[sp]])
    except Exception as e:
        rep["medmcqa"] = f"FAILED {type(e).__name__}: {str(e)[:150]}"
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options")
        for sp in ds:
            rep[f"medqa/{sp}"] = dump("medqa", sp, [dict(x) for x in ds[sp]])
    except Exception as e:
        rep["medqa"] = f"FAILED {type(e).__name__}: {str(e)[:150]}"
    try:
        rows = {"test": [], "validation": [], "dev": []}
        for sub in MMLU_MEDICAL:
            ds = load_dataset("cais/mmlu", sub)
            for sp in rows:
                if sp in ds:
                    rows[sp] += [{**dict(x), "subject": sub} for x in ds[sp]]
        for sp, r in rows.items():
            rep[f"mmlu_medical/{sp}"] = dump("mmlu_medical", sp, r)
    except Exception as e:
        rep["mmlu_medical"] = f"FAILED {type(e).__name__}: {str(e)[:150]}"
    print(json.dumps(rep, indent=1))
    save_report("eval_sets.json", rep)


if __name__ == "__main__":
    main()
