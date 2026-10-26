# Data

## Dataset: The Verdict

- **File:** `raw/the-verdict.txt`
- **Source:** Public domain short story by Edith Wharton
- **Size:** ~20,479 bytes
- **Token count (GPT-2 BPE):** ~5,145 tokens
- **License:** Public domain (pre-1928)

## Usage

This dataset is used for training and validation of Jivi-LLM.

The text is tokenized using the GPT-2 BPE tokenizer via `tiktoken`.
A 90/10 train/val split is applied at the token level.

Sliding-window sequences of length `context_length` are extracted with
configurable stride. Each input sequence is shifted by 1 to produce
the target (next-token prediction).
