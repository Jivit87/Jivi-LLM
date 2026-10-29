# Medical-Domain Jivi-LLM: Research Notes

> Research log for turning Jivi-LLM into a **domain-specific medical GPT (about 124M params, about 500M unique training tokens)**.
> Written 2026-10-27. Reuse this file as the source of truth for data decisions. When something is verified or changed, update it here.
>
> **Confidence tags** used below: **[V]** = read in a primary source during this research. **[S]** = seen only in a search snippet or secondary summary. **[E]** = my estimate or inference. **[?]** = unverified, check before relying on it.
> Not legal advice. Licensing notes are a working summary, not a legal opinion.

---

## 0. TL;DR (decisions this research supports)

1. **Collection is not the bottleneck, selection is.** Open medical text is orders of magnitude larger than 500M tokens (PMC commercial-licensed full text alone is about 159 GB of text [V], so very roughly tens of billions of tokens [E]). The job is to pick the best 500M and to keep the licenses clean.
2. **Primary source: PubMed Central Open Access, commercial-use-allowed subset** (CC0 / CC BY / CC BY-SA / CC BY-ND). Use the pre-built `almanach/Biomed-Enriched` (has an educational-quality score and a commercial split) or `common-pile/pubmed`, instead of crawling PMC ourselves. [V]
3. **Do not use the legacy PMC FTP bulk packages. They were removed in August 2026.** Use the S3 bucket `pmc-oa-opendata` (anonymous, `--no-sign-request`) if we ever need raw PMC. [V]
4. **Train a custom medical BPE tokenizer (about 32k)** instead of GPT-2's 50,257. BioMedLM's 125M ablation gained +1.93 pts on MedQA from a domain tokenizer. [V]
5. **Mix in about 15% general-English "replay" text.** At 500M tokens we cannot learn English from scratch as well as the 30B-token models in the literature, so we need general text for fluency. Start from the mixture in section 6 and tune it with small proxy runs. [E, informed by V sources]
6. **500M unique tokens x up to 4 epochs (about 2B tokens seen)** is the sweet spot. Repetition up to about 4 epochs is nearly as good as fresh data, and 2B is about 80% of the Chinchilla-style 20 tokens/param budget for 124M. [V for the 4-epoch finding, E for the rest]
7. **Set expectations: a 124M model will not pass USMLE-style exams.** The closest prior work (0.1B MedLLM) is near chance on MedQA, but is reasonable on context-grounded QA (PubMedQA). Judge success by held-out medical perplexity plus PubMedQA/MedMCQA, not by medical advice quality. [V]
8. **Open decision for the user:** train from scratch (matches this repo's goal and gives a custom tokenizer) vs initialize from OpenAI GPT-2-124M weights and continue pretraining (probably stronger at 500M tokens, but locks the GPT-2 tokenizer). See section 5.

---

## 1. What "domain-specific" models taught us

### 1.1 Prior work, side by side

| Model | Size | Approach | Medical data | Notes |
|---|---|---|---|---|
| **PubMedBERT** (Microsoft, 2020) | BERT-base | **From scratch**, domain vocab | PubMed abstracts (+ full text in later version) | From-scratch beats continual pretraining of a general model *when the domain has abundant unlabeled text*. General vocab shatters terms: "lymphoma" becomes `l ##ym ##ph ##oma`. [V via Microsoft Research page] |
| **BioGPT** (Microsoft, 2022) | 347M (GPT-2 medium); 1.5B large | From scratch, **in-domain BPE vocab** | about 15M PubMed titles+abstracts (pre-2021) | First widely used biomedical decoder. BioGPT-Large reached 81.0% on PubMedQA after task fine-tuning. [S] |
| **BioMedLM** (Stanford CRFM / MosaicML, 2024) | 2.7B, 32 layers, d=2560, 20 heads, ctx 1024 | **From scratch**, GPT-2 style, custom BPE (28,896 vocab) | PubMed abstracts + full text from The Pile, **34.6B tokens**, 8.67 epochs, 300B tokens seen | 128 A100s, 6.25 days. LR 1.6e-4 cosine, 100-step warmup, batch about 1M tokens, AdamW (0.9, 0.95), bf16. After fine-tuning: MedMCQA 57.3, MedQA 54.7, PubMedQA 74.4, MMLU Med-Genetics 69.0. [V] |
| **Meditron-7B/70B** (EPFL, 2023) | 7B / 70B | **Continued pretraining** of Llama-2 | GAP-Replay: 48.1B tokens | Guidelines + PubMed papers + abstracts + 1% general replay. [V] |
| **MedLLM** (sub-billion, 2026) | **100.29M**, 20 layers, d=512, 8 heads, ctx 512, vocab 32k, RoPE/SwiGLU/RMSNorm | General pretraining (29.5B tokens) then domain pretraining | MedFineWeb: 3B tokens selected from web by embedding similarity to medical QA sets | **Closest to our target.** See 1.3. [V] |

### 1.2 Lessons that apply directly to us

- **Domain tokenizer matters.** BioMedLM: a custom PubMed BPE keeps terms like "thrombin" and "cytotoxicity" as single tokens, and gave +1.93 pts MedQA in a 125M ablation. [V] It also saves embedding parameters (section 4.3).
- **Pure-domain from-scratch works only with lots of tokens.** BioMedLM had 34.6B tokens; we have about 0.5B (70x less). Expect weaker language modeling unless we add general text or start from GPT-2 weights. [E]
- **Replay prevents forgetting.** Meditron put general data at 1% of a 48B-token continued-pretraining mix (420M tokens from RedPajama). [V] That is a *continued-pretraining* number. Our from-scratch case needs far more general text (about 10-15%).
- **Mixing domain data into pretraining beats saving it for the end.** Datology's "Finetuner's Fallacy": for OLMo-1B/3B on 200B tokens of Dolma, mixing in domain data was better than fine-tuning on it afterwards. Best domain fraction depends on token budget: **10% below 30B tokens**, 5% around 50B, 2% beyond 240B. Reaching baseline loss took 1.4x-1.75x fewer tokens. [V]
  - Caveat: that experiment had a *large general* base with a *small domain slice*. Ours is the reverse. The takeaway is "find the mixture by experiment," not "use 10%."
- **Small models split by task type.** MedLLM (0.1B): context-grounded QA (PubMedQA) comes within 2.9 pts of a medically adapted 7B model, while **knowledge-recall QA (MedQA) stays near the floor**. [V] A 124M model has too little capacity to memorize medical facts.
- **Perplexity can improve a lot while accuracy barely moves.** MedLLM's domain fine-tuning halved validation perplexity (27.3 to 13.62) but average accuracy rose only 36.2 to 37.0. The authors call this "capacity reallocation rather than knowledge acquisition." Alignment (DPO) added +1.4 pts. [V]
- **Hallucination is unavoidable.** BioMedLM's authors report wrong numbers (for example, an incorrect vitamin D deficiency rate). Their base model also mimics PubMed style rather than answering questions. [V]

### 1.3 Closest comparable: MedLLM-0.1B (all [V])

- 100.29M params: 20 layers, d_model 512, 8 heads, context 512 (curriculum 128, 256, 512), 32k SentencePiece vocab, tied embeddings (16.4M, 16.3% of params), RMSNorm, RoPE, SwiGLU (FFN 2048).
- Stage 1: 46M docs, about 29.5B tokens (RedPajama-V2, C4, OpenWebText, Wikipedia), 200k steps on 2x H100.
- Stage 2 ("MedFineWeb"): 3B tokens chosen from 2M candidate chunks by embedding similarity (Sentence Transformer + FAISS, mean cosine to top-5 neighbors) against MedMCQA, MedQA and PubMedQA. 25k steps at LR 5e-5. Validation ppl 27.3 to 13.62.
- Results after SFT+DPO: PubMedQA 58.2, MedMCQA 34.9, MedQA 28.1, MMLU 32.4.
- Our setup differs: one-stage from scratch, about 0.5B domain-heavy tokens rather than 29.5B general + 3B medical. We should expect **lower** general fluency than MedLLM unless the mix includes enough general text.

---

## 2. Where the data comes from (and what each source is worth)

### 2.1 Primary: PubMed Central Open Access (full text)

**Licensing groups** [V]: *Commercial use allowed* (CC0, CC BY, CC BY-SA, CC BY-ND), *Non-commercial only* (CC BY-NC, -NC-SA, -NC-ND), and *Other* (no machine-readable CC license, or custom license). Check each article's own license statement.

**Access in 2026 [V]:**
- **Legacy PMC FTP bulk packages, the legacy PMC OA Web Service API and the legacy Cloud Service files were removed the week of 2026-08-24.** Old scripts and tutorials that pull `oa_bulk` packages are broken (a GitHub issue on `bmlibrarian` confirms this [S]).
- Replacement is the **PMC Cloud Service**: S3 bucket `pmc-oa-opendata`, region `us-east-1`, world-readable. Use `https://pmc-oa-opendata.s3.amazonaws.com/` or the AWS CLI with `--no-sign-request`.
- Layout: one prefix per article version, `PMC{PMCID}.{version}/`. Each holds XML (JATS), TXT, PDF, media, and a **per-article JSON metadata object that carries the license**.
- **There are no separate commercial/non-commercial folders.** License filtering must use the JSON metadata or an NCBI E-utilities query with license filters (cc0, cc-by, cc-by-sa, cc-by-nd).
- A daily CSV inventory is documented in the bucket's `README.txt`. It is the only supported way to detect removals. It lags the bucket by up to a day.
- Most articles are version 1. Some have several versions, such as author manuscript plus published version. **Dedupe across versions.**
- The individual PMC OA article directories and packages on FTP are said to be unaffected [S]. We don't need them.

**Pre-built alternatives (recommended, no crawling needed):**

| Dataset | What it is | Size | License handling |
|---|---|---|---|
| `almanach/Biomed-Enriched` [V] | PMC OA paragraphs with metadata | 146M paragraphs: **98.6M commercial**, 47.4M non-commercial | Dataset is CC-BY-4.0 / CC-BY-3.0-IGO. Fields: `text`, `document_type` (study/review/other), `domain`, `section_title`, `educational_score`, `authors`, `article_url`, `license_url`. Multilingual (English, French, Spanish + 6 more). |
| `common-pile/pubmed` [V] | PMC articles converted from nXML to markdown | about 4.07M docs, about 158.9 GB UTF-8 text (45.3 GB files) | Only journals tagged CC BY, CC BY-SA or CC0. Per-document license in `metadata`. |

**Why Biomed-Enriched is first choice:** the `educational_score` field lets us pick the most informative paragraphs to fill a 500M-token budget (the dataset's own paper reports that pretraining on it beats baselines [S]). We can prefer reviews and exclude low-value paragraphs (for example author lists, acknowledgements).
**Gap:** Biomed-Enriched is paragraph-level, so cross-paragraph context is lost unless we regroup by article. Check whether `article_url` + `section_title` lets us reassemble documents. [?]

**How other projects cleaned PMC full text (Meditron, [V]):** removed authors, bibliography, acknowledgements, tables and figures; replaced inline citations with a `[BIB_REF]` token (title truncated to 12 words + first author), figure/table references with `[FIG_REF]`, and equations with `[FORMULA]`; marked section headers as `#` / `##`; prepended the title; normalized whitespace; deduplicated and removed non-English. Their knowledge cutoff was August 2023.

### 2.2 PubMed abstracts

- NLM releases an **annual baseline in December** plus **daily update files**. The 2026 snapshot referenced in the literature = 1,334 baseline files + 98 update files, **about 40.5M citation records, about 28.7M with a non-empty abstract**. [S]
- Rule of thumb from the survey literature: PubMed abstracts about 4.5B words, PMC full text about 13.5B words. [S]
- **Terms [V]:** "NLM freely provides PubMed data. Please note some abstracts may be protected by copyright." Acknowledge NLM as the source. If you redistribute, keep data current or say clearly that it may be out of date. Text mining / AI training is not mentioned either way.
- **Implication:** training on abstracts is common practice (BioGPT, BioMedLM, Meditron all do it), but we should **not redistribute raw abstracts** and should not publish a derived dataset. Keep them in `data/medical/raw/`, which is gitignored.
- Abstracts are dense and clean, but narrow in style. They teach terminology well, and do not teach long-form explanation. Use them as a modest slice (section 6).
- A "PubMed-Scale Dataset of Structured Biomedical Abstracts" (arXiv 2606.11361) exists and may give abstracts pre-split into BACKGROUND / METHODS / RESULTS / CONCLUSIONS. [S, not yet read, worth checking]

### 2.3 Clinical practice guidelines

- Meditron's **GUIDELINES**: 46,469 articles from 16 sources, about 107M training tokens in total. Released subset on Hugging Face (`epfl-llm/guidelines`): **about 35.7K-38K articles from 8 redistributable sources**: CCO, CDC, CMA, ICRC, NICE, SPOR, WHO and WikiDoc. [V]
- Scrapers and cleaning code are public for all 16 sources. [V]
- **License warning [V]:** the HF dataset card lists license "Common Crawl", and some content (Cancer Care Ontario) carries an explicit "may not be reproduced without express written permission" notice. **Treat it as research use, check each source, and keep it out of anything we redistribute.**
- Cleaning they used [V]: removed URLs, references, figures, table delimiters; standardized to section headers + paragraphs; normalized lists; deduplicated by title; dropped very short or non-English articles.
- Value: **high quality, well structured, actionable text** (diagnosis, treatment, dosing), which is closer to what a user would ask about than research papers. Small in volume, so it is a candidate for upsampling (2-4 epochs). The total is capped at about 107M tokens even before license filtering.

### 2.4 Reference and encyclopedic text (small, high quality)

| Source | Content | License / terms | Notes |
|---|---|---|---|
| **MedlinePlus** | Consumer health topics, test info, genetics summaries | **Some content is public domain; other content is copyrighted and licensed only for MedlinePlus** [V]. Public domain: health-topic summaries, medical test info, genetics summaries | Patient-facing language, which PMC lacks. Keep only the public-domain parts. |
| **Wikipedia (medicine)** | Medical articles | CC BY-SA 4.0 [E, standard Wikipedia license, not checked this session] | Use a medicine-category filter. Share-alike applies. Check how that interacts with releasing weights. |
| **NCBI Bookshelf (OA subset)** | Open-access books (over 800 chapters mentioned in OA subset) [S] | Per-book licenses | Check each license before including. **StatPearls is CC BY-NC-ND 4.0** [S], so it is non-commercial and no-derivatives. **Exclude** it from anything we plan to release. |
| **openFDA** | Drug labels / adverse events as JSON | **CC0 / public domain** [S] | Bulk JSON archives. Drug-label text is structured and repetitive. Cap its share. |
| **DailyMed (SPL labels)** | FDA drug labels | Catalog lists **ODbL** [S] | Verify. Highly templated text, so heavy dedup needed. [?] |
| **ClinicalTrials.gov** | Trial records (XML bulk) | Catalog lists ODbL; free to use [S] | Mostly structured fields, low prose value. Probably skip. [?] |
| **medRxiv / bioRxiv** | Preprints (monthly MECA packages with full-text XML/PDF) | Authors consent to TDM; **requester-pays S3 bucket** (`s3://medrxiv-src-monthly`) [S] | Not peer reviewed, and costs money to download. Skip for v1. |

### 2.5 Medical web text (optional)

- **MedFineWeb approach [V]:** select medical-looking pages from a general web corpus by embedding similarity to medical QA text. It brings **consumer / patient-facing** English, which scientific papers lack.
- For us: apply the same idea to **FineWeb-Edu** (already quality-filtered). Use a sentence-embedding model, plus a similarity threshold, to pick maybe 25M tokens.
- Risk: web pages are copyrighted even when the dataset is openly licensed, and quality is uneven (health misinformation). Keep the share small (about 5%) and make it optional.

### 2.6 What we should NOT use

- **MIMIC and other clinical notes:** credentialed access and data-use agreements. No redistribution. Not for a hobby model. [E]
- **Non-commercial PMC articles (CC BY-NC*):** research-only. Using them restricts whether we can ever publish the weights. Keep them out of the default mix. They could form an optional "research-only" shard later.
- **PMC-Patients** (CC BY-NC-SA 4.0) [S]: non-commercial.
- **Paywalled textbooks / UpToDate / exam banks:** copyrighted. (MedQA's book corpus is the same issue.)
- **Benchmark test sets** (MedQA, MedMCQA, PubMedQA test splits). See decontamination in section 3.

### 2.7 Licensing policy (recommended)

- **Default policy: train only on the commercial-use-allowed subset + public-domain sources.** That keeps the option of publishing the checkpoint.
- Store the **license of every document** in its metadata and carry it through every pipeline stage. Keep `data/medical/manifest/licenses.csv` per source.
- CC BY needs attribution. Publish a `SOURCES.md` in the model card listing sources and licenses.
- CC BY-SA (share-alike) and ODbL (share-alike on databases): whether they reach a trained model's weights is not settled. Note the risk. If it matters, put those sources in a separate shard and train with and without them.
- Disclose the model is **not for clinical use**.

---

## 3. Data pipeline design

### 3.1 Stages

```
01 download   -> data/medical/raw/<source>/             (immutable, checksummed)
02 extract    -> data/medical/interim/<source>/*.jsonl  (one doc per line + metadata)
03 clean      -> text normalization, boilerplate / reference removal
04 filter     -> language, length, quality score, license, date
05 dedupe     -> exact hash, then MinHash near-dup (within and across sources)
06 decontam   -> remove docs that overlap eval sets  -> data/medical/filtered/
07 tokenizer  -> train custom BPE on a sample of the filtered data
08 tokenize   -> data/medical/tokenized/{train,val}/*.bin  (uint16 shards)
09 report     -> token counts per source, mixture check -> data/medical/reports/
```

Every stage should be **resumable and idempotent**, write only to its own folder, and log counts in and out.

### 3.2 Document record format (`jsonl`)

```json
{"id": "pmc:PMC123456.1:p07", "source": "pmc_oa", "text": "...",
 "title": "...", "year": 2021, "license": "CC BY 4.0", "license_url": "...",
 "lang": "en", "doc_type": "review", "quality": 3.4, "n_chars": 5120}
```

### 3.3 Cleaning rules (from Meditron, Biomed-Enriched and common practice)

- Keep title, abstract, body paragraphs, section headers. Drop authors, affiliations, references, acknowledgements, funding, conflict-of-interest and ethics boilerplate.
- Tables / figures: drop them, or keep captions only. Replace citations like `[12]` / `(Smith et al., 2019)` with a single marker or remove them. Replace formulas with a placeholder.
- Normalize unicode and whitespace, fix mojibake, drop control characters.
- Drop very short docs (for example under 100 tokens) and docs that are mostly numbers or tables.
- Add `<|endoftext|>` between documents. For the packed-sequence format, never let a sample span unrelated docs without it.
- **De-identify:** PMC case reports are already published under consent, but run a simple scan for emails, phone numbers and URLs and strip them.
- Consider dropping **retracted articles** [?: check whether PMC metadata exposes retraction status].

### 3.4 Deduplication

- Exact dedup by content hash first. Then MinHash-LSH near-dup (for example 5-gram shingles, Jaccard about 0.8).
- Dedupe **across sources**: an abstract is often contained in its full-text article. Keep the full text and drop the matching abstract.
- Dedupe across PMC versions (author manuscript vs final).
- Drug labels (DailyMed / openFDA) are extremely repetitive. Dedupe aggressively or cap by source.
- Meditron deduplicated by title matching and dropped non-English. [V]

### 3.5 Decontamination (do this before training, not after)

- Evaluate on MedQA, MedMCQA, PubMedQA (and MMLU medical subsets). **Remove any training document that overlaps their test sets.**
- Method used in recent work [S]: a two-stage pipeline: flag docs sharing **8-gram** overlap with benchmark prompts, then remove those with high normalized alignment (catches paraphrases while ignoring incidental overlap).
- Published contamination levels after cleaning were 0.08-0.24% (below a 0.3% "negligible" threshold). [S]
- Practical note: **PubMedQA is built from PubMed abstracts**, so its test contexts probably sit in our PubMed data. Decontaminate by PMID if possible. [E]
- Keep the reference sets in `data/medical/eval/` and treat that folder as read-only.

### 3.6 Splits

- **Validation by whole document** (never split an article between train and validation). Hold out about 5-10M tokens, **stratified by source**. Keep a second small "latest year" slice, to measure drift on newer papers. [E]
- Also keep a **general-English validation** slice (for example a few Gutenberg books or held-out FineWeb-Edu), to watch fluency and forgetting.
- Final test sets are the public benchmarks, never mixed into any training or validation shard.

### 3.7 Storage format

- Tokenized shards as **`uint16` `.bin` files** with `np.memmap` (works while vocab < 65,536, which holds for a 32k BPE). Shard size about 100M tokens (200 MB). Plus a small `meta.json` (vocab size, tokenizer hash, per-source token counts, dtype).
- Keep a per-source offset index so we can resample sources with different weights without re-tokenizing.
- The existing loader (`train_gpu.py`) loads a whole int64 tensor with `torch.load`. That works at 25M tokens and will not scale: 500M tokens as int64 = 4 GB, as `uint16` = 1 GB. Switch to memmap shards. [E]

---

## 4. Tokenizer

### 4.1 Why change it

- GPT-2's BPE (50,257 tokens) was trained on web text. Medical terms split into many pieces, which wastes sequence length and gives the model more subword assembly work. [V for the general point]
- BioMedLM trained its own BPE on PubMed abstracts (vocab 28,896, `min_frequency=2`) and saw **+1.93 pts on MedQA** in a 125M-parameter ablation. [V]

### 4.2 Recommendation

- Train a **byte-level BPE, vocab about 32,000** (pad to 32,000 or 32,768 for GPU efficiency) with Hugging Face `tokenizers`, on a **representative sample of the final mixture** (including the general replay slice, so everyday English isn't fragmented). Include `<|endoftext|>`. [E]
- Measure **tokens per word** and **tokens per medical term** on held-out text vs GPT-2's tokenizer. Record both in `reports/`.
- Save it in `data/medical/tokenizer/` with a hash, and store the hash inside every checkpoint.

### 4.3 What changes in the repo

- `src/jivi_llm/tokenizer.py` currently hard-wires `tiktoken.get_encoding("gpt2")`, so it needs a loader for the custom tokenizer. `checkpoint.py` records `"tokenizer": "gpt2"`, and the configs use `vocab_size: 50257`.
- **Parameter count:** the embedding matrix shrinks by (50,257 - 32,000) x 768 = **14.0M params per matrix**. With tied weights, a 12-layer / 768-dim model goes from about 124M to about **110M**. Untied (the repo's current lm_head) it is about 135M. To land back near 124M with a 32k vocab and tied weights, use **14 layers** (about 124.6M). [E, arithmetic checked]
- Tying input and output embeddings is worth doing here: it saves 24.6M params at 32k vocab. [E]

---

## 5. Open decision: from scratch vs GPT-2 initialization

| | From scratch (custom tokenizer) | Continue from GPT-2-124M weights |
|---|---|---|
| Matches repo goal ("built from scratch") | Yes | Less so |
| Tokenizer | Custom medical 32k | Must keep GPT-2 50k |
| Needs to learn English from 0 | **Yes, using about 15% of a 500M budget** | No, already learned from WebText |
| Expected quality at 500M tokens | Lower fluency, better medical vocab efficiency | Likely better overall at this data size [E] |
| Evidence | PubMedBERT / BioMedLM did it with billions of tokens (only when abundant domain text) [V] | Meditron / continued pretraining; a 125M model gained +8.1% MMLU and +7.6% HellaSwag from 400M-1B tokens of domain continued pretraining (educational domain) [S] |
| Config compatibility | Free to change | GPT-2 weights use `qkv_bias=True`; this repo's configs have `qkv_bias: false` [V from repo files] |

**Recommendation:** make from-scratch the main run (it is the project's point), and run a **GPT-2-init continued-pretraining baseline** on the same data as a control. The comparison on held-out medical perplexity and PubMedQA is itself a useful result. The user should confirm this.

---

## 6. Proposed 500M-token budget (starting hypothesis, to be tuned)

Targets are **unique tokens**. With up to 4 epochs we see about 2B tokens in training. Small high-quality sources get repeated more (still at or under 4 epochs).

| # | Source | Share | Unique tokens | Notes |
|---|---|---|---|---|
| 1 | PMC OA full text (commercial license, educational_score filter, English) | **55%** | about 275M | Backbone. Prefer reviews and high `educational_score`. |
| 2 | PubMed abstracts (not already in PMC full text) | **12%** | about 60M | Terminology density. Dedupe against #1. |
| 3 | Clinical guidelines (8 redistributable sources) | **5%** | about 25M | Upsample. Check per-source terms. |
| 4 | Reference text: MedlinePlus (PD parts), Wikipedia medicine, Bookshelf OA, openFDA | **8%** | about 40M | Patient-facing + encyclopedic. Cap drug-label share. |
| 5 | Medical web text (FineWeb-Edu filtered by embedding similarity) | **5%** | about 25M | Optional. Consumer language. |
| 6 | General-English replay (FineWeb-Edu / Wikipedia) | **15%** | about 75M | Fluency + anti-forgetting. |

Rationale and caveats:
- Percentages are **hypotheses, not findings.** The Datology result says the best domain fraction changes with token budget, and Meditron's 1% replay was for continued pretraining. [V] So **run a proxy sweep**: train a small model (for example the 30M config) on about 100M-token subsets with replay at 5% / 15% / 30%, and compare on medical perplexity and a general-English validation set. Pick the mixture from that.
- Ablation worth running: **with vs without the abstracts slice**, and **with vs without the web slice**.
- If a source comes up short (guidelines, in particular), do not pad it by repeating beyond 4 epochs. Move the share to PMC.

### 6.1 Compute sanity check [E]

- 124M params, Chinchilla-style 20 tokens/param -> about 2.5B tokens. Our plan: 500M unique x 4 epochs = 2B tokens, about 80% of that. Muennighoff et al. found that up to about 4 epochs of repeated data gives negligible loss change vs unique data at a fixed compute budget, and that returns decay to zero with more repetition. [V] (Their runs went up to 900B tokens and 9B params.)
- Throughput on this machine (RTX PRO 4000 Blackwell, 24 GB) is not measured yet. **Measure tokens/sec on a short run before committing** to a 2B-token job. Earlier rough guess in chat: on the order of a day for 2.5B tokens, so treat as unverified.

---

## 7. Evaluation plan

### 7.1 Metrics

1. **Held-out perplexity per source** (PMC, abstracts, guidelines, general English). This is the primary training signal.
2. **PubMedQA** (500 test questions; yes/no/maybe; context-grounded). Closest to what a small model can do. Chance for 3 classes is about 33%, but the label distribution is skewed, so compare against the majority-class baseline. [E]
3. **MedMCQA** (4,183 dev questions, 4 options, chance 25%).
4. **MedQA** (1,273 test, 4-option version, chance 25%). A 124M model will likely sit near chance. That is expected (MedLLM-0.1B got 28.1). [V]
5. **MMLU medical subsets** (clinical knowledge, anatomy, college medicine, medical genetics, professional medicine, college biology).
6. General-capability smoke test (for example HellaSwag or a small English validation set), to detect forgetting. Meditron-style smoke tests used MMLU-Pro, IFEval and ARC-Challenge for that purpose. [V]

### 7.2 Method notes

- Score multiple-choice by **log-probability of each option** (the "top-token" approach Meditron uses, following the HELM implementation). [V] Zero-shot and few-shot, no fine-tuning, as the first look.
- Benchmark fine-tuning (as BioMedLM and Meditron did) measures *adaptability*, not what pretraining learned. Report zero-shot and fine-tuned numbers separately.
- Baselines to include: untrained random init (sanity), GPT-2-124M zero-shot, GPT-2-124M + continued pretraining on our data, our from-scratch model.
- Reference numbers from the literature for scale: MedLLM-0.1B (after SFT+DPO): PubMedQA 58.2 / MedMCQA 34.9 / MedQA 28.1. BioMedLM-2.7B (fine-tuned): 74.4 / 57.3 / 54.7. [V]
- Decontaminate (section 3.5) before reporting any benchmark number.
- Sample generations are **qualitative only**. Do not treat fluent output as correct. Always attach a "not medical advice" disclaimer.

---

## 8. Folder structure (created under `data/medical/`)

The existing Gutenberg data in `data/raw/` and `data/processed/` is untouched (`train_gpu.py` still reads it).

```
data/medical/
├── README.md              # what's here, how to regenerate it
├── manifest/              # TRACKED in git: small, describes every source
│   ├── sources.yaml       #   name, URL, license, access method, date fetched, version
│   └── licenses.csv       #   per-source license + redistribution notes
├── raw/                   # IGNORED: immutable downloads, one folder per source
│   ├── pmc_oa/
│   ├── pubmed_abstracts/
│   ├── guidelines/
│   ├── reference/         #   medlineplus, wikipedia_med, bookshelf_oa, openfda
│   └── general_replay/    #   fineweb_edu, wikipedia
├── interim/               # IGNORED: extracted text as jsonl, one folder per source
├── filtered/              # IGNORED: after clean + dedupe + decontamination
├── tokenizer/             # tokenizer files + hash (small; decide whether to track)
├── tokenized/             # IGNORED: uint16 shards: train/, val/, meta.json
├── eval/                  # IGNORED: benchmark sets (read-only, used for decontamination)
│   ├── pubmedqa/ medmcqa/ medqa/ mmlu_medical/
└── reports/               # TRACKED: token counts per source, mixture, tokenizer stats
```

Pipeline code goes in a separate `data_pipeline/` folder at the repo root (numbered scripts 01 to 09 as in 3.1), not inside `data/`.

Disk budget [E]: raw + interim + filtered + tokenized probably 30-120 GB depending on how much of PMC we stream. Free space is 423 GB.

---

## 9. Risks and open questions

| Risk / question | Why it matters | Next step |
|---|---|---|
| From scratch vs GPT-2 init | Largest effect on final quality | User decision (section 5) |
| License strictness (CC BY-SA, ODbL, NC sources) | Whether the weights can be published | Default to commercial-allowed + public domain; keep SA/ODbL in a separable shard |
| PubMed abstract copyright | "Some abstracts may be protected by copyright" [V] | Train on them, do not redistribute |
| Biomed-Enriched is paragraph-level | May lose document context | Check if articles can be reassembled from `article_url` + `section_title` [?] |
| Exact token counts per source | The budget table is estimated | Stream a small sample of each source and measure with the final tokenizer |
| PMC JSON metadata fields (license, retraction) | Needed for filtering | Download a few `PMC*.json` and inspect [?] |
| Releasable guideline tokens | Could be well under 25M | Count after license filtering |
| DailyMed / ClinicalTrials.gov license | Catalog snippets conflict with "public domain" expectations | Read the terms pages directly [?] |
| Knowledge cutoff / outdated medicine | Guidelines and drug advice change | Keep `year` metadata, report the cutoff date in the model card |
| Safety | Model will hallucinate doses, numbers, diagnoses | Model card disclaimer, no clinical use, don't build a "medical advice" demo |
| Compute time | Not measured | Benchmark tokens/sec on a short run first |

### Not yet researched (candidates for a second pass)

- Exact PMC OA commercial-subset token count (so we know the pool we are selecting from).
- Whether synthetic/rephrased data (WRAP-style rephrasing, QA from guidelines) helps at this scale. (Fully Open Meditron used about 386K synthetic examples, about 100M tokens, but for fine-tuning, not pretraining. [V])
- Curriculum / sequence-length warmup (MedLLM used 128, 256, 512). Could speed up early training.
- Quality classifiers other than `educational_score` (for example a small fastText classifier trained on our own labels).
- Instruction-tuning data for after pretraining (MedQA/MedMCQA/PubMedQA train splits, guideline-derived QA).

---

## 10. Sources

**Papers and project pages**
- BioMedLM: https://arxiv.org/abs/2403.18421 (HTML: https://arxiv.org/html/2403.18421v1)
- PubMedBERT (Microsoft Research): https://www.microsoft.com/en-us/research/publication/domain-specific-language-model-pretraining-for-biomedical-natural-language-processing/
- BioGPT (summary pages): https://the-decoder.com/biogpt-is-a-microsoft-language-model-trained-for-biomedical-tasks/
- Meditron-70B: https://arxiv.org/abs/2311.16079 (tables read directly from the PDF, pages 3-8)
- MedLLM sub-billion: https://arxiv.org/abs/2607.27490
- Fully Open Meditron: https://arxiv.org/abs/2605.16215
- Scaling Data-Constrained Language Models (Muennighoff et al.): https://arxiv.org/abs/2305.16264
- The Finetuner's Fallacy (Datology): https://www.datologyai.com/blog/finetuners-fallacy
- Domain-adaptive continued pretraining of small LMs (125M): https://arxiv.org/abs/2504.09687
- Biomed-Enriched: https://arxiv.org/abs/2506.20331
- CMR scaling law (critical mixture ratio): https://arxiv.org/abs/2407.17467
- Structured PubMed abstracts dataset: https://arxiv.org/abs/2606.11361 (not read)
- Marin "Medical Data Anneal" report (not read; redirect not followed): https://wandb.ai/marin-community/marin/reports/Medical-Data-Anneal--VmlldzoxMjk4NjE0Mw

**Data access and licensing**
- PMC FTP service notice (legacy files removed): https://pmc.ncbi.nlm.nih.gov/tools/ftp
- PMC Cloud Service (AWS): https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
- AWS Open Data registry entry: https://github.com/awslabs/open-data-registry/blob/main/datasets/ncbi-pmc.yaml
- PMC OA subset license groupings: https://pmc.ncbi.nlm.nih.gov/tools/openftlist
- PubMed baseline terms: https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/README.txt
- Biomed-Enriched dataset: https://huggingface.co/datasets/almanach/Biomed-Enriched
- Common Pile PubMed: https://huggingface.co/datasets/common-pile/pubmed
- Meditron guidelines dataset: https://huggingface.co/datasets/epfl-llm/guidelines
- openFDA terms: https://open.fda.gov/terms
- MedlinePlus copyright: https://medlineplus.gov/copyright.html
- bioRxiv / medRxiv TDM: https://www.biorxiv.org/tdm
- DailyMed SPL downloads: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm

---

## 11. Collection log (2026-10-28): what was actually collected

Scripts are in `data_pipeline/` (numbered 00 to 08). Data is under `data/medical/filtered/`. Token counts are **estimates** (chars / 4.2); exact counts come after the custom tokenizer is trained.

| Source | Docs | Est. tokens | Share | Plan | Notes |
|---|---|---|---|---|---|
| PMC OA full text (commercial licenses) | 35,169 | **279.9M** | 56.7% | 55% | Top articles by educational score from 7 of 26 shards (every 4th). 15% reviews, 3.8% clinical cases, rest studies. |
| PubMed abstracts | 146,322 | **60.7M** | 12.3% | 12% | 20 baseline files (n1000 to n1323). Only abstracts with no publisher (c) line or a CC licence. |
| Guidelines: CDC + WikiDoc | 25,487 | **63.8M** | 12.9% | 5% | WikiDoc is 53M of this. Over plan: can be down-weighted at sampling time. |
| Reference: openFDA + MedlinePlus | 1,743 | **7.2M** | 1.5% | 8% | Under plan. Wikipedia medicine and Bookshelf OA not collected yet. |
| General English replay (FineWeb-Edu) | 57,872 | **81.9M** | 16.6% | 15% | Score >= 3.2. The dataset's own count says 75M tokens. |
| Medical web slice | 0 | 0 | 0% | 5% | Not collected. Optional. |
| **Total** | | **about 493M** | | about 500M | |

Cleaning (`08_dedupe_decontam.py`, 13-word shingles): 39 documents removed as benchmark-contaminated (PMC 8, guidelines 26, replay 4, abstracts 1), about 35 exact duplicates, and only 2 abstracts overlapped the PMC text.
Benchmarks held out for decontamination and evaluation (in `data/medical/eval/`): PubMedQA (1,000), MedMCQA (test 6,150 / validation 4,183), MedQA (test 1,273), MMLU-medical (8 subjects, 1,775 questions).

### Findings that update earlier sections

- **Biomed-Enriched can be reassembled into articles.** Paragraphs are contiguous and ordered by `article_id`, so full documents are rebuilt by grouping. Resolves the open question in section 9. Shards are chronological by PMC id, so every 4th shard samples evenly across eras.
- **The educational score is tightly bunched at article level**: median 4.08, p90 4.22, p99 4.42. The final cutoff of about 4.31 (studies) keeps roughly the top 7% of text in the sampled shards.
- **Clinical cases score lower** (median 3.95), so a global ranking squeezes them out. Reserved shares were needed: only 10.7M tokens (3.8%) of cases pass the 3.9 floor.
- **PubMed abstracts: 60% of recent abstracts carry a publisher copyright line.** The conservative filter keeps about 15 to 40% depending on the file. `--allow-copyrighted` relaxes it (grey area). Almost none duplicate the PMC sample.
- **Guideline licences (checked 2026-10-28):** WHO is CC BY-NC-SA 3.0 IGO (excluded). NICE's licence says AI-use requests are not covered (excluded). WikiDoc is CC BY-SA 3.0. CDC is mostly public domain. CCO restricts reproduction (excluded). CMA, ICRC, SPOR unchecked (held back, about 6M tokens).
- **Network is the bottleneck: about 3.5 to 4 MB/s total, whatever the host.** Parallel streams do not help. The Hugging Face xet downloader stalled, so downloads use resumable `curl`. A full 49 GB PMC pull would take about 3.5 hours, so we sampled 7 shards (about 13 GB).
- **openFDA partition 1 had only 730 unique generic drugs** (20,000 labels). More partitions would add more drugs.

### Still to do before training

1. Decide the mixture weights (WikiDoc is heavy, reference is light). Weights are applied when sampling, not by deleting data.
2. Train the custom 32k BPE tokenizer, then tokenize to `uint16` shards and get exact token counts.
3. Optional gaps: Wikipedia medicine, NCBI Bookshelf OA, a medical web slice, more openFDA partitions, CMA/ICRC/SPOR licence checks.
4. Hold out validation by document, by source.

---

## 12. Tokenizer and exact token counts (2026-10-28)

Tokenizer: 32,000-vocab byte-level BPE (`data/medical/tokenizer/med_bpe_32k.json`, `<|endoftext|>` = id 0, sha256 `c5dcece7...`), trained on a 1.5 GB weighted sample (validation docs excluded). Scripts `09_train_tokenizer.py`, `10_tokenize.py`. Tokenized data: `data/medical/tokenized/<source>/{train,val}.bin` (uint16) and `tokenized/meta.json`.

| Source | Train tokens | Val tokens | Docs (train / val) |
|---|---|---|---|
| PMC OA | 234.2M | 4.67M | 34,475 / 694 |
| PubMed abstracts | 51.0M | 1.01M | 143,465 / 2,857 |
| Guidelines (CDC + WikiDoc) | 57.0M | 1.24M | 24,977 / 510 |
| Reference (openFDA + MedlinePlus) | 6.0M | 0.12M | 1,707 / 36 |
| General replay (FineWeb-Edu) | 75.7M | 1.65M | 56,692 / 1,180 |
| **Total** | **423.9M** | **8.69M** | |

- **Exact total is 432.6M tokens, about 14% below the 500M goal.** My chars/4.2 estimate in section 11 overshot: medical text tokenizes at about 4.9 characters per token with the custom BPE, not 4.2.
- **Tokenizer vs GPT-2** on held-out text: PMC 10% fewer tokens, abstracts and guidelines 6% fewer, reference 11% fewer, general English 4% more. Rare terms are single tokens (thrombocytopenia: 1 vs 7, glomerulonephritis: 1 vs 6, acetyltransferase: 1 vs 4).
- Round-trip check passed: decode(encode(doc)) equals the NFC-normalized original.
- Validation is about 2% of documents per source, chosen by a stable hash of the document id.

### Ways to close the gap to 500M (about +67M tokens)
- **Cheapest:** raise the PMC budget. The 263K candidates from the 7 shards are still on disk (`interim/pmc_oa`, 8.2 GB), so re-running `07` with a target of about 358M estimated tokens lowers the cutoff from about 4.31 to about 4.25. No new download. Then re-run `08` and `10`.
- Relax the abstract copyright filter (`--allow-copyrighted`), a grey area.
- Download more PMC shards (about 9 min each) for a bigger candidate pool.
- Add the gaps: Wikipedia medicine, Bookshelf OA, more openFDA partitions, a medical web slice.

### Architecture note for the 124M target
`GPTModel` has separate input and output embeddings (no weight tying). With a 32k vocab, 768 wide, that gives about 135M params at 12 layers. Options: tie the embeddings and use 14 layers (about 124.6M), or stay untied with 10 layers (about 121M) or 11 layers (about 128M).

### 12b. Update after raising the PMC budget (2026-10-28, later)

Re-ran `07` with `--target-tokens 358e6` (quality cutoff for studies about 4.29, 43,862 articles, 43,851 after cleaning), then `08` and `10`. Tokenizer unchanged (same sha).

| Source | Train tokens | Val tokens |
|---|---|---|
| PMC OA | 303.4M | 5.90M |
| General replay | 75.7M | 1.65M |
| Guidelines (CDC + WikiDoc) | 57.0M | 1.24M |
| PubMed abstracts | 51.0M | 1.01M |
| Reference | 6.0M | 0.12M |
| **Total** | **493.1M** | **9.92M** (503.0M overall) |

- Mix now: PMC 61.5%, replay 15.4%, guidelines 11.6%, abstracts 10.3%, reference 1.2% (of train tokens). Weights are still applied at sampling time.
- Clean-up this run: PMC contamination removals 11 (benchmark overlap), no other changes.

### Model changes made (2026-10-28)
- `GPTConfig.tie_weights` (default False) and tying in `GPTModel`; `configs/med_124m.yaml` = 32k vocab, 14 layers, d=768, 12 heads, ctx 1024, tied: **124.56M params**.
- `get_tokenizer()` accepts a path to a custom `tokenizer.json`; `save_checkpoints(..., tokenizer=...)` records it. Tests: 25 pass (6 new in `tests/test_med_domain.py`).
- Still to do: a medical training script (`train_med.py`) that memory-maps `tokenized/<source>/train.bin`, samples by mix weights, and evaluates per-source validation perplexity.

---

## 13. Training run `med124m_v1` (2026-10-28 to 10-29)

Script `train_med.py`, config `configs/med_124m.yaml` (124.56M params, 14 layers, d=768, tied embeddings, ctx 1024), fused attention + `torch.compile` (max-autotune-no-cudagraphs).

- **Settings:** 1.5B tokens (3.0 passes over the 493.1M-token train corpus), 131,072 tokens/step, 11,444 steps, peak LR 6e-4 (chosen from a 3-way pilot: 3e-4 gave val 5.79, 6e-4 gave 5.67, 1e-3 gave 6.00 at 30M tokens), 300-step warmup, flat until 80%, cosine decay to 6e-5, AdamW (0.9, 0.95), weight decay 0.1, grad clip 1.0, dropout 0, bf16 autocast, GPT-2 style init (std 0.02, residual projections scaled by 1/sqrt(2*layers)). Sources sampled in proportion to size (so each source got about 3 epochs).
- **Speed:** 68,300 tokens/s sustained, 6.17 hours total. No loss spikes in the whole run; grad norm settled near 0.23 to 0.3.
- **Final validation loss (nats per token, custom tokenizer):** mix 2.756 (perplexity about 15.7); PMC 2.671 (14.5); guidelines 2.537 (12.6); abstracts 2.730 (15.3); reference 1.682 (5.4); general English 3.362 (28.8). Progress of the mix loss: 4.69 (step 500), 3.31 (2,000), 3.05 (4,500), 2.93 (7,750), 2.89 (9,500, decay starts at 9,155), 2.76 (11,444).
- **Overfitting:** none seen; train loss about 2.6 to 2.8 vs val 2.76 at the end.
- **Generated samples at the end:** fluent, clinical in style. Metformin is now described as an oral anti-diabetic agent (the earlier sulfonylurea error is gone) but with an invented mechanism ("inhibits glycolysis in muscle cells"); the pneumonia prompt names *Streptococcus pneumoniae*; the chronic-kidney-disease prompt produced made-up research-paper phrasing ("supported by our results"). The general-English prompt drifts to genes. **Fluent but not factually reliable. Not for medical use.**
- **Checkpoints:** `checkpoints/med124m_v1/step_{001000..011444}.pt` (1.5 GB each, include optimizer state) and `latest.pt`. A weights-only export for inference would be about 0.5 GB.
- Perplexities are per token of the custom 32k tokenizer, so they are not comparable with GPT-2-tokenizer numbers.

### Not done yet
Held-out benchmark evaluation (PubMedQA, MedMCQA, MedQA, MMLU-medical) with a random-init and a GPT-2-124M baseline; a weights-only export; sampling from the final model with different prompts/temperatures; optional further training on more unique data (see section 12 notes).

---

## 14. Evaluation of `med124m_v1` (2026-10-29)

Script `eval_med.py`, results in `data/medical/reports/eval_results.json`. Zero-shot, no few-shot examples, no fine-tuning. Baselines: chance (25% for 4-option) and OpenAI GPT-2-124M. Multiple choice scored two ways: *cloze* (likelihood of the answer text; `acc_norm` divides by length) and *letter* (likelihood of A/B/C/D after listing the options). 95% confidence interval is about +/-2.4 (MedQA), +/-1.4 (MedMCQA), +/-2.3 (MMLU), +/-3.1 (PubMedQA) points.

| Benchmark (chance) | Medical model: cloze / cloze_norm / letter | GPT-2 124M: cloze / cloze_norm / letter |
|---|---|---|
| MedQA 4-option, n=1273 (25%) | 25.3 / 27.3 / 27.7 | 22.2 / 25.8 / 27.7 |
| MedMCQA validation, n=4183 (25%) | 23.5 / 28.4 / 32.0 | 21.1 / 26.2 / 31.9 |
| MMLU medical, n=1561 (25%) | 25.5 / 30.8 / 22.8 | 24.5 / 27.5 / 22.6 |
| PubMedQA, n=1000 (majority "yes" = 55.2%) | 43.3 (418 yes / 582 no / 0 maybe) | 52.2 (724 yes / 276 no / 0 maybe) |

**Bits per byte on held-out text (lower is better; fair across tokenizers):**

| Source | Medical model | GPT-2 | Change |
|---|---|---|---|
| PMC full text | 0.706 | 0.917 | -23% |
| PubMed abstracts | 0.777 | 0.955 | -19% |
| Guidelines | 0.608 | 0.927 | -34% |
| Reference (drug labels, MedlinePlus) | 0.581 | 0.784 | -26% |
| General English | 1.036 | 0.960 | +8% (worse) |

**Reading:** the model models medical *language* far better than GPT-2 (19 to 34% fewer bits per byte) and general English a bit worse. On exam-style questions it is **close to chance**, with only marginal, mostly within-noise edges over GPT-2 (cloze_norm: MedQA +1.5, MedMCQA +2.2, MMLU +3.3). The MedMCQA letter score (32%) is the same for both models, so it is a letter-bias artifact, not knowledge. PubMedQA is below the always-"yes" baseline for both models, and the medical model (43.3%) is worse than GPT-2 (52.2%): it leans "no" and never says "maybe". This matches the literature (a 0.1B medical model is near the floor on MedQA, MedLLM section 1.3): at this size, knowledge-recall and reasoning questions need far more capacity.

### What could raise the numbers
Fine-tuning on the benchmarks' own training sets (MedQA 10K, MedMCQA 182K, PubMedQA labelled/artificial) is the standard route; MedLLM-0.1B reached about MedMCQA 34.9 and PubMedQA 58.2 after SFT + preference tuning. More unique pretraining data and a larger model are the routes to real knowledge.
