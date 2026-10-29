# Jivi-LLM

**A GPT-style language model written from scratch in PyTorch, and Jivi-Med-124M, a 124.6M-parameter medical-domain model trained with it.**

Everything here is built by hand: the tokenizer pipeline, attention, transformer blocks, training loop, checkpointing, decoding, the data collection and cleaning pipeline, and the evaluation harness. One GPU, no training frameworks.

> **Not for medical use.** Jivi-Med-124M is a small research model. It writes fluent clinical-sounding text, but it often states things that are false. Do not use it for medical advice, diagnosis, or decisions. See [Limitations](#limitations-and-intended-use).

## Highlights

- **From scratch.** Multi-head causal attention, pre-LayerNorm transformer blocks, AdamW training loop, resumable checkpoints, temperature / top-k decoding. 30 unit tests.
- **A medical-domain model, end to end.** 493M tokens of collected, license-checked, deduplicated and benchmark-decontaminated medical text; a custom 32k BPE tokenizer; 1.5B training tokens in 6.2 hours on a single 24 GB GPU.
- **Fast on modest hardware.** Fused attention, `torch.compile` and bf16 take training from 17.6k to 68k tokens/s on an RTX PRO 4000 Blackwell.
- **Honest evaluation.** Zero-shot benchmarks with confidence intervals against chance and against OpenAI's GPT-2 124M, plus bits per byte on held-out text so models with different tokenizers can be compared fairly.

## Jivi-Med-124M

### Model

| | |
|---|---|
| Parameters | **124.56M** (input and output embeddings tied) |
| Layers / hidden size / heads | 14 / 768 / 12 (head size 64) |
| Feed-forward | 3072, GELU |
| Context length | 1024 tokens |
| Vocabulary | 32,000, byte-level BPE trained on medical + general text |
| Normalization / positions | pre-LayerNorm / learned absolute |
| Attention | fused causal attention (`F.scaled_dot_product_attention`) |
| Config | [`configs/med_124m.yaml`](configs/med_124m.yaml) |

### Training data

493.1M training tokens (9.9M held out for validation, split by whole document). Sources were chosen for commercially usable or public-domain terms, with two caveats: PubMed abstracts are kept only when they carry no publisher copyright notice (or a CC license), and FineWeb-Edu is ODC-By while the web pages inside it keep their own copyrights.

| Source | Train tokens | Share | License / terms |
|---|---|---|---|
| PubMed Central open-access full text (top-quality articles, via [Biomed-Enriched](https://huggingface.co/datasets/almanach/Biomed-Enriched)) | 303.4M | 61.5% | CC BY and related commercial-use licenses |
| General English ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)) | 75.7M | 15.4% | ODC-By |
| Clinical guidelines (CDC, WikiDoc) | 57.0M | 11.6% | US-government public domain; CC BY-SA 3.0 |
| PubMed abstracts (only those without a publisher copyright line, or under a CC license) | 51.0M | 10.3% | NLM terms |
| openFDA drug labels, MedlinePlus topics | 6.0M | 1.2% | CC0; public domain |

Guidelines from WHO (non-commercial license), NICE (license excludes AI use) and Cancer Care Ontario (no reproduction) were left out. About 40 documents that overlapped the benchmark test sets were removed. How the data was chosen and cleaned, with every decision and measurement, is in [`docs/med_domain.md`](docs/med_domain.md).

### Training

| | |
|---|---|
| Tokens | 1.5B (3.0 passes over the corpus), 11,444 steps of 131,072 tokens |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on weight matrices |
| Learning rate | 6e-4 peak (chosen from a 3-way pilot), 300-step warmup, flat until 80%, cosine decay to 6e-5 |
| Regularization | none (dropout 0), gradient clipping at 1.0 |
| Precision / compile | bf16 autocast, `torch.compile` (max-autotune) |
| Hardware / time | 1x NVIDIA RTX PRO 4000 Blackwell (24 GB), 6.17 hours at 68.3k tokens/s |
| Result | validation loss 2.756 nats/token (perplexity 15.7); no loss spikes, no overfitting |

### Results

Zero-shot, no fine-tuning, no few-shot examples. Multiple choice is scored by the likelihood of each answer text (length-normalized). Chance is 25% for the first three rows; always answering "yes" on PubMedQA scores 55.2%. 95% confidence intervals are about ±1.4 to ±3 points.

| Benchmark | Jivi-Med-124M | GPT-2 124M |
|---|---|---|
| MedQA, USMLE 4-option (1,273 q) | 27.3 | 25.8 |
| MedMCQA, validation (4,183 q) | 28.4 | 26.2 |
| MMLU medical, 8 subjects (1,561 q) | 30.8 | 27.5 |
| PubMedQA (1,000 q) | 43.3 | 52.2 |

Bits per byte on held-out text (lower is better; comparable across tokenizers):

| Text | Jivi-Med-124M | GPT-2 124M | Change |
|---|---|---|---|
| PMC full text | 0.706 | 0.917 | 23% lower |
| PubMed abstracts | 0.777 | 0.955 | 19% lower |
| Clinical guidelines | 0.608 | 0.927 | 34% lower |
| Drug labels, MedlinePlus | 0.581 | 0.784 | 26% lower |
| General English | 1.036 | 0.960 | 8% higher |

**Reading the results.** The model models medical text much better than a general model of the same size. It does **not** answer exam questions: it is within a few points of chance, and its edge over GPT-2 is mostly inside the error bars. On PubMedQA it leans toward "no" and falls below always answering "yes". This matches published work on sub-billion medical models, where knowledge-recall questions stay near chance. Fine-tuning on the benchmarks' training sets, more unique data, or a larger model are the routes to improve it. Details: [`docs/med_domain.md`](docs/med_domain.md#14-evaluation-of-med124m_v1-2026-10-29).

Sample (temperature 0.7, top-k 50, prompt in bold):

> **Aspirin should not be given to children because** of the potential risks. ### Precautions - Aspirin should not be given to patients with severe renal insufficiency because of the risk of bleeding…

Fluent and in the style of a drug label, but note it never mentions Reye syndrome. This is typical: style is learned long before facts.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q          # 30 tests
```

Model weights are not in the repository (a checkpoint is 1.5 GB with optimizer state). To use the model, train it (below) or place a checkpoint at `checkpoints/med124m_v1/latest.pt`.

```bash
python generate_med.py --prompt "Metformin is" --max-new 80 --temperature 0.7 --top-k 50
python eval_med.py --models med gpt2               # benchmarks + bits per byte, writes data/medical/reports/eval_results.json
```

## Reproduce

The medical corpus is built by numbered scripts in [`data_pipeline/`](data_pipeline/). Downloads are large and slow on a normal connection (about 13 GB for the PMC shards).

```bash
python data_pipeline/00_fetch_pmc.py --shards 0 4 8 12 16 20 24 --tau-edu 4.15   # PMC full text (calls 01_pmc_select.py)
python data_pipeline/02_guidelines.py                                             # CDC + WikiDoc guidelines
python data_pipeline/03_pubmed_abstracts.py --files 1000 1017 1034 1051 1068 1085 1102 1119 1136 1153 \
                                                    1170 1187 1204 1221 1238 1255 1272 1289 1306 1323
python data_pipeline/04_general_replay.py --tokens 75e6                           # FineWeb-Edu
python data_pipeline/05_reference.py                                              # openFDA + MedlinePlus
python data_pipeline/06_fetch_eval.py                                             # benchmark sets (held out)
python data_pipeline/07_finalize_pmc.py --target-tokens 358e6                     # pick the best PMC articles
PYTHONHASHSEED=0 python data_pipeline/08_dedupe_decontam.py                       # dedupe + remove benchmark overlap
python data_pipeline/09_train_tokenizer.py                                        # 32k medical BPE
python data_pipeline/10_tokenize.py                                               # uint16 token shards

python train_med.py --name med124m_v1 --tokens 1.5e9 --lr 6e-4 --warmup 300 --decay-frac 0.2 \
                    --dropout 0.0 --compile max --eval-every 250 --save-every 1000
```

`train_med.py` resumes with `--resume`, and stops cleanly when you create `checkpoints/<name>/STOP`.

## Repository layout

```
src/jivi_llm/       the model library: config, tokenizer, dataset, attention, transformer block, model,
                    loss, evaluate, train, checkpoint, generate
configs/            model configs: med_124m.yaml (medical), gpt_124m.yaml, small_30m.yaml, debug.yaml
data_pipeline/      numbered scripts that build the medical corpus (00 to 10)
data/medical/       manifest (sources, licenses), reports, tokenizer; raw/filtered/tokenized data is git-ignored
train_med.py        medical-model training (memory-mapped shards, source mixing, resumable)
eval_med.py         benchmarks and bits-per-byte evaluation
generate_med.py     text generation from a checkpoint
train_gpu.py        earlier training script (Project Gutenberg experiments)
tests/              30 unit tests
docs/               research notes and plans (see below)
Resources/          reference material used while building the model
```

## Documentation

| Document | What's in it |
|---|---|
| [`docs/med_domain.md`](docs/med_domain.md) | The research behind the medical model, source licenses, the full data-collection log, tokenizer, training run, and evaluation |
| [`docs/improve.md`](docs/improve.md) | Improvement plan and the diagnosis of the first training runs |
| [`data/README.md`](data/README.md), [`data/medical/README.md`](data/medical/README.md) | Corpus descriptions and data folder layout |

## Limitations and intended use

- **It states false things confidently.** Examples seen in testing: an invented mechanism for metformin, fabricated trial statistics, a diagnosis that drifts to an unrelated disease. A 124M model learns the *style* of medical writing long before it learns facts.
- **Near chance on exam-style questions** (see Results). It is a research artifact, not a tool.
- **English only, and a snapshot.** Training text runs up to 2025; medical guidance changes.
- **Biased toward its sources.** Mostly open-access research papers, so it favors research phrasing and the topics well covered there. It was not evaluated for demographic bias.
- **Intended use:** studying how domain-specific pretraining works at small scale; a base for further fine-tuning experiments. **Not intended for** clinical use, patient-facing applications, or anywhere an incorrect medical statement could cause harm.

## Earlier experiments: Project Gutenberg

Before the medical corpus, the model was developed on 115 public-domain books (25.6M tokens; whole-book validation). These runs used one NVIDIA RTX PRO 4000 Blackwell with bf16, AdamW and cosine decay, and are single runs with one seed.

| Config | Params | Tokens seen | Time | Val loss | Val perplexity |
|---|---|---|---|---|---|
| `small_30m` | 49.3M | 30M | 6 min | 4.53 | ~93 |
| `gpt_124m` | 163M (untied) | 30M | 32 min | 4.28 | ~72 |
| `small_30m` | 49.3M | 100M | 20 min | **4.16** | ~64 |

With that small corpus, more passes mattered more than model size, which motivated collecting a much larger, domain-specific one. `train_gpu.py` and `download_data.py` reproduce these runs.

## License

No license has been chosen for the code yet. Data used for training keeps its own license (table above). Sources were chosen for commercially usable or public-domain terms; see the caveats in the training-data section.

## Citation

```bibtex
@misc{jivi_llm_2026,
  title        = {Jivi-LLM: a GPT built from scratch, and Jivi-Med-124M},
  author       = {Jivit87},
  year         = {2026},
  howpublished = {\url{https://github.com/Jivit87/Jivi-LLM}}
}
```

## Acknowledgements

Training data comes from PubMed Central and PubMed (US National Library of Medicine), Biomed-Enriched (Touchent, Godey, de la Clergerie), the Meditron clinical-guidelines release (EPFL), FineWeb-Edu (Hugging Face), openFDA, MedlinePlus, CDC and WikiDoc. Evaluation uses PubMedQA, MedMCQA, MedQA and MMLU.
