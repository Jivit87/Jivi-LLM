"""
Tokenize the cleaned corpus with the custom medical BPE into flat uint16 binaries, one train/val pair per source:

    data/medical/tokenized/<source>/train.bin
    data/medical/tokenized/<source>/val.bin
    data/medical/tokenized/meta.json      (exact token counts, vocab size, EOT id, tokenizer hash)

Every document is followed by the <|endoftext|> token (id 0). Validation is a deterministic ~2% of documents per source
(see common.is_val), so no validation document is ever in training. Memory-map the .bin files at training time.

Usage: python data_pipeline/10_tokenize.py
"""
import hashlib
import json
import os
import sys

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MED, SOURCE_DIRS, is_val, iter_source, save_report  # noqa: E402

BATCH_DOCS = 2000


def flush(tok, texts, fout, eot):
    if not texts:
        return 0
    n = 0
    for enc in tok.encode_batch(texts):
        ids = np.array(enc.ids + [eot], dtype=np.uint16)
        fout.write(ids.tobytes())
        n += len(ids)
    return n


def main():
    tpath = os.path.join(MED, "tokenizer", "med_bpe_32k.json")
    tok = Tokenizer.from_file(tpath)
    assert tok.get_vocab_size() < 65536, "uint16 cannot hold this vocab"
    eot = tok.token_to_id("<|endoftext|>")
    sha = hashlib.sha256(open(tpath, "rb").read()).hexdigest()
    meta = {"vocab_size": tok.get_vocab_size(), "eot_id": eot, "dtype": "uint16", "tokenizer_sha256": sha,
            "val_percent": 2, "sources": {}}
    for src in SOURCE_DIRS:
        outdir = os.path.join(MED, "tokenized", src)
        os.makedirs(outdir, exist_ok=True)
        counts = {"train_tokens": 0, "val_tokens": 0, "train_docs": 0, "val_docs": 0}
        bufs = {"train": [], "val": []}
        with open(os.path.join(outdir, "train.bin"), "wb") as ftr, open(os.path.join(outdir, "val.bin"), "wb") as fva:
            outs = {"train": ftr, "val": fva}

            def drain(split):
                counts[f"{split}_tokens"] += flush(tok, bufs[split], outs[split], eot)
                counts[f"{split}_docs"] += len(bufs[split])
                bufs[split] = []

            for d in iter_source(src):
                split = "val" if is_val(d["id"]) else "train"
                bufs[split].append(d["text"])
                if len(bufs[split]) >= BATCH_DOCS:
                    drain(split)
            drain("train")
            drain("val")
        meta["sources"][src] = counts
        print(f"{src}: {counts}", flush=True)
    meta["total_train_tokens"] = sum(c["train_tokens"] for c in meta["sources"].values())
    meta["total_val_tokens"] = sum(c["val_tokens"] for c in meta["sources"].values())
    with open(os.path.join(MED, "tokenized", "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    save_report("tokenized_counts.json", meta)
    print(f"TOTAL train {meta['total_train_tokens']/1e6:.1f}M, val {meta['total_val_tokens']/1e6:.2f}M tokens", flush=True)

    # sanity: decode a few documents back and compare with the NFC-normalized original
    import unicodedata

    src = "pmc_oa"
    arr = np.fromfile(os.path.join(MED, "tokenized", src, "val.bin"), dtype=np.uint16)
    first = arr[: int(np.where(arr == eot)[0][0])]
    decoded = tok.decode(first.tolist())
    for d in iter_source(src):
        if is_val(d["id"]):
            ok = decoded == unicodedata.normalize("NFC", d["text"])
            print("round-trip first val doc identical:", ok, flush=True)
            break


if __name__ == "__main__":
    main()
