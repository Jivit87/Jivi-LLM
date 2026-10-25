# Jivi-LLM

A from-scratch educational and reproducible implementation of a GPT-2-style decoder-only Transformer (~124M parameters) in PyTorch.

## Overview

**Jivi-LLM** is built block-by-block following principles from *Build a Large Language Model (From Scratch)* by Sebastian Raschka and reference decoding strategies.

### Goals
- Implement every component (tokenization, embeddings, causal multi-head attention, feed-forward, layer normalization, residual blocks) from scratch.
- Random initialization — no pretrained weights loaded.
- Next-token pretraining pipeline with cross-entropy loss, AdamW, evaluation, and checkpoint/resume support.
- Configurable decoding strategies: greedy, temperature scaling, top-$k$, and multinomial sampling.

## Project Structure

```text
jivi-llm/
├── PROJECT_PLAN.md             # Detailed roadmap and milestones
├── requirements.txt            # Core dependencies
├── configs/                    # Model configurations (debug & gpt_124m)
├── data/                       # Raw and processed datasets (ignored in git)
├── checkpoints/                # Model checkpoints (ignored in git)
├── logs/                       # Training logs (ignored in git)
├── Resources/                  # Reference material & notebooks
├── src/                        # Core implementation modules
└── tests/                      # Component tests
```

## Getting Started

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the complete roadmap and milestones.
