# Data

## Corpus: Project Gutenberg books

- **Script:** `download_data.py` (downloads, strips the Gutenberg licence header/footer, normalizes blank lines, writes `raw/combined.txt`)
- **Source:** Project Gutenberg plain-text ebooks, 115 distinct books (fiction by Austen, Dickens, Twain, the Brontes, Tolstoy, Conan Doyle, Wells and others, plus some essays, philosophy and poetry)
- **License:** public domain
- **Size:** about 94 MB, **25.6M GPT-2 BPE tokens** (`combined.txt`)
- **Tokenizer:** GPT-2 BPE via `tiktoken`
- **Files:** one `raw/<name>.txt` per book (original text, including the header), plus `raw/combined.txt`. Both are gitignored, so run `download_data.py` to recreate them.

A few "Project Gutenberg" mentions remain inside the book texts. They were not removed.

## Train / validation split

Validation is by whole book, so no validation text appears in training:

| Split | Books | Tokens |
|---|---|---|
| Train | the other 107 books | 24.6M |
| Validation | emma, dracula, the-verdict, moby-dick, treasure-island, heart-of-darkness, walden, jane-eyre | 1.48M |

`combined.txt` is not used for this split, because cutting it at a fixed fraction would make validation the end of the last book. The split was done in throwaway experiment scripts, not in the repo's code yet.

## Is it enough?

About 25M tokens suits models up to roughly 30-50M parameters (a few epochs). The 124M model would need around 2.5B tokens to be well trained, so it needs a larger corpus (for example FineWeb-Edu or OpenWebText).

## Earlier dataset

The first pipeline tests used `raw/the-verdict.txt` (~20 KB, ~5,145 tokens), a short story by Edith Wharton. It is now one of the validation books.
