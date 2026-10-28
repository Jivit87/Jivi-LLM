"""
Small, high-quality reference sources:
  - MedlinePlus health topics (NLM; the topic summaries are public domain), consumer-friendly language
  - openFDA drug labels (CC0): one partition, de-duplicated by generic name, capped so templated label text stays small

Usage: python data_pipeline/05_reference.py [--fda-partition 1] [--fda-cap-tokens 10e6]
"""

import argparse
import html
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTERIM, RAW, est_tokens, fetch, normalize_ws, save_report, write_jsonl  # noqa: E402

TAG = re.compile(r"<[^>]+>")
LI = re.compile(r"</?(li|p|br|h\d|ul|ol|div)[^>]*>", re.I)


def html_to_text(s):
    s = LI.sub("\n", s or "")
    return normalize_ws(html.unescape(TAG.sub("", s)))


def medlineplus(date="2026-10-27"):
    path = fetch(f"https://medlineplus.gov/xml/mplus_topics_{date}.xml",
                 os.path.join(RAW, "reference", f"mplus_topics_{date}.xml"))
    root = ET.parse(path).getroot()
    chars = 0

    def docs():
        nonlocal chars
        for t in root.iter("health-topic"):
            if t.get("language") != "English":
                continue
            title = t.get("title")
            summ = html_to_text(t.findtext("full-summary"))
            if len(summ) < 400:
                continue
            text = f"# {title}\n\n{summ}"
            chars += len(text)
            yield {"id": f"medlineplus:{t.get('id')}", "source": "reference_medlineplus",
                   "license": "US-gov public domain (health topic summaries)", "title": title, "text": text}

    n = write_jsonl(os.path.join(INTERIM, "reference", "medlineplus.jsonl"), docs())
    print(f"medlineplus: {n} topics ~{est_tokens(chars)/1e6:.2f}M tokens", flush=True)
    return {"docs": n, "est_tokens": est_tokens(chars)}


SECTIONS = [("indications_and_usage", "Indications and usage"), ("dosage_and_administration", "Dosage and administration"),
            ("contraindications", "Contraindications"), ("warnings", "Warnings"),
            ("warnings_and_cautions", "Warnings and precautions"), ("precautions", "Precautions"),
            ("adverse_reactions", "Adverse reactions"), ("drug_interactions", "Drug interactions"),
            ("use_in_specific_populations", "Use in specific populations"), ("overdosage", "Overdosage"),
            ("description", "Description"), ("clinical_pharmacology", "Clinical pharmacology"),
            ("mechanism_of_action", "Mechanism of action")]


def fda(partition, cap_tokens):
    name = f"drug-label-{partition:04d}-of-0014.json.zip"
    path = fetch(f"https://download.open.fda.gov/drug/label/{name}", os.path.join(RAW, "reference", name))
    with zipfile.ZipFile(path) as z:
        data = json.load(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8"))
    best = {}  # generic name -> (n_sections_chars, doc)
    for r in data["results"]:
        of = r.get("openfda") or {}
        gen = (of.get("generic_name") or [None])[0]
        if not gen or "HUMAN PRESCRIPTION DRUG LABEL" not in " ".join(r.get("spl_product_data_elements", [""]) or [""]) and \
                "HUMAN PRESCRIPTION" not in (of.get("product_type") or [""])[0].upper():
            continue
        parts, size = [], 0
        for key, head in SECTIONS:
            if r.get(key):
                body = html_to_text(" ".join(r[key]))
                body = re.sub(r"^\s*\d+(\.\d+)*\s+[A-Z][A-Z &,/()-]{4,}\s+", "", body)  # "1 INDICATIONS AND USAGE" prefix
                if len(body) > 80:
                    parts.append(f"## {head}\n\n{body}")
                    size += len(body)
        if len(parts) < 4:
            continue
        brand = (of.get("brand_name") or [""])[0]
        title = gen.title() + (f" ({brand.title()})" if brand and brand.lower() != gen.lower() else "")
        key = gen.lower()
        if key not in best or size > best[key][0]:
            best[key] = (size, {"id": f"fda:{r.get('set_id')}", "source": "reference_openfda", "license": "CC0 (openFDA)",
                                "title": title, "text": f"# {title}\n\n" + "\n\n".join(parts)})
    docs, total = [], 0
    for size, d in sorted(best.values(), key=lambda x: -x[0]):
        if total >= cap_tokens:
            break
        docs.append(d)
        total += est_tokens(len(d["text"]))
    write_jsonl(os.path.join(INTERIM, "reference", f"openfda_{partition:04d}.jsonl"), docs)
    print(f"openfda: {len(docs)} unique drugs ~{total/1e6:.1f}M tokens (from {len(data['results'])} labels, "
          f"{len(best)} unique generics)", flush=True)
    os.remove(path)
    return {"docs": len(docs), "est_tokens": total, "unique_generics": len(best)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fda-partition", type=int, default=1)
    ap.add_argument("--fda-cap-tokens", type=float, default=10e6)
    a = ap.parse_args()
    rep = {"medlineplus": medlineplus(), "openfda": fda(a.fda_partition, a.fda_cap_tokens)}
    save_report("reference_extract.json", rep)


if __name__ == "__main__":
    main()
