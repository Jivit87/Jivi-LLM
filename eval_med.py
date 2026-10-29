"""
Zero-shot evaluation of the medical model (and a GPT-2-124M baseline) on medical benchmarks + held-out text.

Benchmarks (data/medical/eval/, made by data_pipeline/06_fetch_eval.py):
  PubMedQA (1000 labelled, yes/no/maybe given the abstract), MedMCQA validation (4183), MedQA USMLE 4-option test (1273),
  MMLU medical subjects test (1561).
Multiple choice is scored two ways:
  cloze  : which option text is most likely after "Question: ...\nAnswer:" (acc = total log-prob, acc_norm = per character)
  letter : which of " A"/" B"/" C"/" D" is most likely after the question with all options listed
Held-out text: bits per byte on validation documents, so models with different tokenizers are comparable.
No fine-tuning and no few-shot examples. Small models are often near chance here; that is the point of the baselines.

Usage: python eval_med.py --models med gpt2 [--quick 100]
"""
import argparse
import ast
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "src")
sys.path.insert(0, "data_pipeline")
EVAL = "data/medical/eval"


class LM:
    def __init__(self, kind, ckpt):
        self.kind = kind
        if kind == "med":
            from jivi_llm.config import GPTConfig
            from jivi_llm.model import GPTModel
            from jivi_llm.tokenizer import get_tokenizer

            ck = torch.load(ckpt, map_location="cpu")
            cfg = GPTConfig(**ck["config"])
            tok = get_tokenizer(ck["tokenizer"])
            model = GPTModel(cfg)
            model.load_state_dict(ck["model_state"])
            self.fn, self.ctx, self.eot, self._enc = model.cuda().eval(), cfg.context_length, tok.eot_token, tok.encode
            self.name = f"medical 124M (step {ck['step']})"
        else:
            import tiktoken
            from transformers import GPT2LMHeadModel

            model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().eval()
            enc = tiktoken.get_encoding("gpt2")
            self.fn = lambda x: model(x).logits
            self.ctx, self.eot = 1024, 50256
            self._enc = lambda t: enc.encode(t, disallowed_special=())
            self.name = "GPT-2 124M (OpenAI, general web text)"

    def encode(self, text):
        return self._enc(text)


@torch.no_grad()
def score(lm, pairs, max_tokens=20000):
    """pairs: [(ctx_ids, cont_ids)] -> [(sum log-prob of cont given ctx, n_cont_tokens)]."""
    items = []
    for idx, (c, k) in enumerate(pairs):
        seq = c + k
        if len(seq) > lm.ctx:  # keep the end (the question and options), drop the oldest tokens
            cut = len(seq) - lm.ctx
            seq, start = seq[cut:], max(len(c) - cut, 1)
        else:
            start = len(c)
        items.append((idx, seq, start))
    items.sort(key=lambda t: -len(t[1]))
    out, i = [None] * len(pairs), 0
    while i < len(items):
        L = len(items[i][1])
        batch = items[i:i + max(1, max_tokens // L)]
        i += len(batch)
        x = torch.zeros(len(batch), L, dtype=torch.long)
        for r, (_, seq, _) in enumerate(batch):
            x[r, :len(seq)] = torch.tensor(seq)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = lm.fn(x.cuda())
        for r, (idx, seq, start) in enumerate(batch):
            lp = torch.log_softmax(logits[r, start - 1:len(seq) - 1].float(), -1)
            tgt = torch.tensor(seq[start:], device="cuda")[:, None]
            out[idx] = (lp.gather(1, tgt).sum().item(), len(seq) - start)
    return out


def as_obj(x):
    """Fields are real lists/dicts in the saved jsonl; accept their string form too."""
    return ast.literal_eval(x) if isinstance(x, str) else x


def load(name, split, n=None):
    rows = [json.loads(l) for l in open(f"{EVAL}/{name}/{split}.jsonl")]
    return rows[:n] if n else rows


def ci(correct, n):
    p = correct / n
    return p, 1.96 * math.sqrt(p * (1 - p) / n)


def run_mc(lm, qs, label):
    """qs: [{'prompt_cloze': str, 'prompt_letter': str, 'options': [str], 'gold': int}] -> accuracy dict."""
    letters = [lm.encode(f" {c}") for c in "ABCD"]
    assert all(len(t) == 1 for t in letters), "letters must be single tokens"
    cloze_pairs, letter_pairs = [], []
    for q in qs:
        c = lm.encode(q["prompt_cloze"])
        cloze_pairs += [(c, lm.encode(" " + o)) for o in q["options"]]
        p = lm.encode(q["prompt_letter"])
        letter_pairs += [(p, t) for t in letters[:len(q["options"])]]
    cs, ls = score(lm, cloze_pairs), score(lm, letter_pairs)
    k = [len(q["options"]) for q in qs]
    acc = norm = let = 0
    ci_, li_ = 0, 0
    for q, nk in zip(qs, k):
        a, ci_ = cs[:nk], ci_ + nk
        b, li_ = ls[:nk], li_ + nk
        cs, ls = cs[nk:], ls[nk:]
        acc += max(range(nk), key=lambda j: a[j][0]) == q["gold"]
        norm += max(range(nk), key=lambda j: a[j][0] / max(1, len(q["options"][j]))) == q["gold"]
        let += max(range(nk), key=lambda j: b[j][0]) == q["gold"]
    n = len(qs)
    return {"n": n, "cloze_acc": ci(acc, n), "cloze_acc_norm": ci(norm, n), "letter_acc": ci(let, n)}


def tasks(n):
    mmlu = []
    for d in load("mmlu_medical", "test", n):
        ch, subj = as_obj(d["choices"]), d["subject"].replace("_", " ")
        opts = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", ch))
        mmlu.append({"prompt_cloze": f"The following is a question about {subj}.\n\nQuestion: {d['question']}\nAnswer:",
                     "prompt_letter": f"The following is a multiple choice question about {subj}.\n\n{d['question']}\n{opts}\nAnswer:",
                     "options": ch, "gold": int(d["answer"])})
    mcqa = []
    for d in load("medmcqa", "validation", n):
        ch = [d["opa"], d["opb"], d["opc"], d["opd"]]
        opts = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", ch))
        mcqa.append({"prompt_cloze": f"Question: {d['question']}\nAnswer:",
                     "prompt_letter": f"Question: {d['question']}\n{opts}\nAnswer:", "options": ch, "gold": int(d["cop"])})
    medqa = []
    for d in load("medqa", "test", n):
        o = as_obj(d["options"])
        ch = [o[c] for c in "ABCD"]
        opts = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", ch))
        medqa.append({"prompt_cloze": f"Question: {d['question']}\nAnswer:",
                      "prompt_letter": f"Question: {d['question']}\n{opts}\nAnswer:", "options": ch,
                      "gold": "ABCD".index(d["answer_idx"])})
    return {"MedQA (USMLE, 4-option)": medqa, "MedMCQA (validation)": mcqa, "MMLU medical (8 subjects)": mmlu}


def run_pubmedqa(lm, n):
    rows = load("pubmedqa", "train", n)
    labels = ["yes", "no", "maybe"]
    pairs = []
    for d in rows:
        ctx = as_obj(d["context"])["contexts"]
        prompt = lm.encode(f"Context: {' '.join(ctx)}\nQuestion: {d['question']}\nAnswer:")
        pairs += [(prompt, lm.encode(" " + l)) for l in labels]
    sc = score(lm, pairs)
    correct, pred_dist = 0, {l: 0 for l in labels}
    for i, d in enumerate(rows):
        s = [x[0] for x in sc[3 * i:3 * i + 3]]
        p = labels[max(range(3), key=lambda j: s[j])]
        pred_dist[p] += 1
        correct += p == d["final_decision"]
    gold = [d["final_decision"] for d in rows]
    maj = max(labels, key=gold.count)
    return {"n": len(rows), "acc": ci(correct, len(rows)), "majority": (maj, gold.count(maj) / len(rows)),
            "pred_dist": pred_dist}


@torch.no_grad()
def bits_per_byte(lm, docs_per_source=120):
    from common import SOURCE_DIRS, is_val, iter_source

    res = {}
    for src in SOURCE_DIRS:
        nll = nbytes = 0.0
        k = 0
        for d in iter_source(src):
            if not is_val(d["id"]):
                continue
            text = d["text"][:3000]
            ids = [lm.eot] + lm.encode(text)
            ids = ids[:lm.ctx]
            (lp, n), = score(lm, [(ids[:1], ids[1:])])
            nll += -lp
            nbytes += len(text.encode("utf-8")) if len(ids) < lm.ctx else len(text.encode("utf-8")) * (len(ids) / lm.ctx)
            k += 1
            if k >= docs_per_source:
                break
        res[src] = nll / math.log(2) / nbytes
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["med", "gpt2"])
    ap.add_argument("--ckpt", default="checkpoints/med124m_v1/latest.pt")
    ap.add_argument("--quick", type=int, default=None, help="only the first N items of each benchmark")
    a = ap.parse_args()
    results = {}
    for kind in a.models:
        t0 = time.time()
        lm = LM(kind, a.ckpt)
        print(f"\n=== {lm.name}", flush=True)
        r = {"name": lm.name, "benchmarks": {}}
        for name, qs in tasks(a.quick).items():
            r["benchmarks"][name] = run_mc(lm, qs, name)
            b = r["benchmarks"][name]
            print(f"{name:28s} n={b['n']:5d}  cloze {b['cloze_acc'][0]*100:5.1f}  cloze_norm {b['cloze_acc_norm'][0]*100:5.1f}  "
                  f"letter {b['letter_acc'][0]*100:5.1f}  (+/-{b['cloze_acc_norm'][1]*100:.1f})", flush=True)
        pq = run_pubmedqa(lm, a.quick)
        r["benchmarks"]["PubMedQA (yes/no/maybe)"] = pq
        print(f"{'PubMedQA':28s} n={pq['n']:5d}  acc {pq['acc'][0]*100:5.1f} (+/-{pq['acc'][1]*100:.1f})  majority '{pq['majority'][0]}' "
              f"{pq['majority'][1]*100:.1f}  predictions {pq['pred_dist']}", flush=True)
        r["bits_per_byte"] = bits_per_byte(lm, 20 if a.quick else 120)
        print("bits per byte (lower = better):", {k: round(v, 3) for k, v in r["bits_per_byte"].items()}, flush=True)
        results[kind] = r
        del lm
        torch.cuda.empty_cache()
        print(f"({time.time()-t0:.0f}s)", flush=True)
    out = "data/medical/reports/eval_results.json" if not a.quick else "data/medical/reports/eval_results_quick.json"
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
