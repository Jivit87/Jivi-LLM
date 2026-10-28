# Medical corpus

Research and decisions live in `../../med_domain.md`. Read it first.

Target: about 500M unique tokens for a ~124M GPT. Mixture and licenses: `manifest/`.

| Folder | Tracked in git? | Purpose |
|---|---|---|
| `manifest/` | yes | sources, licenses, versions |
| `reports/` | yes | token counts per source, tokenizer stats |
| `tokenizer/` | yes | custom BPE + hash |
| `raw/` | no | immutable downloads, one folder per source |
| `interim/` | no | extracted text (jsonl, one doc per line + metadata) |
| `filtered/` | no | cleaned, deduplicated, decontaminated |
| `tokenized/` | no | `uint16` shards: `train/`, `val/`, `meta.json` |
| `eval/` | no | benchmark sets, read-only, used for decontamination |

Pipeline stages: download, extract, clean, filter, dedupe, decontaminate, tokenizer, tokenize, report.
Not medical advice. Do not use for clinical decisions.
