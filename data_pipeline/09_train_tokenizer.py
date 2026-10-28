"""
Train a 32k byte-level BPE tokenizer on a weighted sample of the cleaned medical corpus (validation docs excluded), then
compare it with GPT-2's tokenizer on held-out text and on medical terms.

Usage: python data_pipeline/09_train_tokenizer.py --sample-chars 1.5e9
Output: data/medical/tokenizer/med_bpe_32k.json, data/medical/reports/tokenizer_report.json
"""
import argparse
import hashlib
import json
import os
import random
import sys

import tiktoken
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MED, SOURCE_DIRS, REPORTS, FILTERED, is_val, iter_source, save_report  # noqa: E402

EOT = "<|endoftext|>"
# share of the tokenizer-training sample per source (general English is included so everyday words get merges)
MIX = {"pmc_oa": 0.50, "pubmed_abstracts": 0.12, "guidelines": 0.10, "reference": 0.03, "general_replay": 0.25}
TERMS = ["lymphoma", "acetyltransferase", "thrombocytopenia", "cytotoxicity", "hypertension", "methotrexate",
         "immunohistochemistry", "electrocardiogram", "glomerulonephritis", "SARS-CoV-2", "randomized", "metformin",
         "atrial fibrillation", "pneumonia", "carcinoma", "pharmacokinetics", "hemoglobin", "myocardial infarction",
         "polymerase chain reaction", "anticoagulant", "the", "children", "therefore"]


def source_chars(source):
    return sum(len(d["text"]) for d in iter_source(source))


def sample_texts(budget, seed=0):
    rng = random.Random(seed)
    for src, share in MIX.items():
        total = source_chars(src)
        p = min(1.0, share * budget / total)
        got = 0
        for d in iter_source(src):
            if is_val(d["id"]) or rng.random() > p:
                continue
            got += len(d["text"])
            yield d["text"]
        print(f"  sample {src}: {got/1e6:.0f}M chars (p={p:.2f})", flush=True)


def build(vocab_size):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # GPT-2 style regex split
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    return tok, trainer


def evaluate(tok, n_docs=150, seed=1):
    gpt2 = tiktoken.get_encoding("gpt2")
    rng = random.Random(seed)
    res = {}
    for src in MIX:
        chars = mine = base = 0
        k = 0
        for d in iter_source(src):
            if not is_val(d["id"]) or rng.random() > 0.3:
                continue
            t = d["text"][:20000]
            chars += len(t)
            mine += len(tok.encode(t).ids)
            base += len(gpt2.encode(t, allowed_special="all"))
            k += 1
            if k >= n_docs:
                break
        res[src] = {"docs": k, "tokens_per_1k_chars_medbpe": round(mine / chars * 1000, 1),
                    "tokens_per_1k_chars_gpt2": round(base / chars * 1000, 1),
                    "medbpe_vs_gpt2": round(mine / base, 3)}
    terms = {t: {"medbpe": len(tok.encode(" " + t).ids), "gpt2": len(gpt2.encode(" " + t))} for t in TERMS}
    return res, terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-chars", type=float, default=1.5e9)
    ap.add_argument("--vocab-size", type=int, default=32000)
    a = ap.parse_args()
    tok, trainer = build(a.vocab_size)
    print("training...", flush=True)
    tok.train_from_iterator(sample_texts(a.sample_chars), trainer=trainer)
    assert tok.token_to_id(EOT) == 0, "EOT should be id 0"
    outdir = os.path.join(MED, "tokenizer")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "med_bpe_32k.json")
    tok.save(path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    res, terms = evaluate(tok)
    rep = {"vocab_size": tok.get_vocab_size(), "eot_id": 0, "sha256": sha, "sample_chars": a.sample_chars, "mix": MIX,
           "held_out": res, "terms": terms}
    save_report("tokenizer_report.json", rep)
    print(json.dumps({"held_out": res, "sha256": sha[:16]}, indent=1))
    print("terms (medbpe vs gpt2 token counts):", {t: (v["medbpe"], v["gpt2"]) for t, v in terms.items()})


if __name__ == "__main__":
    main()
