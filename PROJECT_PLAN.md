# Jivi-LLM — From-Scratch GPT Pretraining Project

**Project status:** Planning and source audit  
**Model target:** GPT-2-style decoder-only Transformer, approximately 124 million parameters  
**Initial development hardware:** MacBook Pro M3, 8 GB unified memory, 512 GB storage  
**Later training hardware:** Remote NVIDIA RTX 4090 (24 GB VRAM), when the data scale and experiment justify it  
**Primary reference:** *Build a Large Language Model (From Scratch)* by Sebastian Raschka  
**Starting code reference:** `LLM Decoding Strategies.ipynb`  
**Project name:** **Jivi-LLM**

---

## 1. Project mission

Jivi-LLM is an educational, reproducible project to understand and implement a GPT-style language model block by block. We will start with an architecture we implement ourselves, initialize its weights randomly, train it with next-token prediction, evaluate what it learns, and save checkpoints so experiments can be resumed.

The aim is not to wrap an existing pretrained model or to build a production chatbot. The aim is to learn how a decoder-only Transformer works and to make a small, verifiable pretraining pipeline that can later scale to a larger text corpus and remote GPU.

### Explicit scope

**Included**
- Implementing the GPT-style architecture from scratch in PyTorch.
- Text preparation, tokenization, input/target construction and data loading.
- Embeddings, causal self-attention, multi-head attention and Transformer blocks.
- Random initialization of all model parameters.
- Next-token pretraining with cross-entropy loss and AdamW.
- Training/validation measurements, generated samples and perplexity where appropriate.
- Reproducible configurations, tests, logging, checkpoint saving and resuming.
- Text generation for evaluating the pretrained base model.

**Explicitly excluded**
- Loading pretrained GPT-2 weights.
- Instruction fine-tuning or supervised chat fine-tuning.
- Production inference infrastructure, hosted APIs, deployment and serving.
- Claiming that a model is a useful general-purpose assistant merely because training completes.

The result of this scope is a **pretrained base language model experiment**, not a ChatGPT-equivalent assistant. It may learn corpus-specific patterns and produce text, but its quality will depend heavily on the amount, diversity, cleanliness and suitability of the training data, as well as the training compute.

---

## 2. Source material and how we will use it

### 2.1 The book

The book is the conceptual and implementation roadmap. Its chapters map naturally to our project:

| Book section | What it contributes to Jivi-LLM | Project output |
|---|---|---|
| Chapter 1 — Understanding LLMs | Decoder-only GPT overview, next-token pretraining, Transformer concepts and scale considerations | A clear design specification and vocabulary |
| Chapter 2 — Working with text data | Tokenization, BPE, token IDs, special tokens, sliding-window sampling, embeddings and positional embeddings | Tokenizer/data pipeline and tested batches |
| Chapter 3 — Attention mechanisms | Query/key/value projections, attention scores, causal masking, dropout and multi-head attention | Attention modules with shape and masking tests |
| Chapter 4 — GPT model from scratch | Layer normalization, GELU feed-forward layers, residual/shortcut connections, stacked Transformer blocks, final normalization and output head | Complete randomly initialized GPT model |
| Chapter 5 — Pretraining on unlabeled data | Train/validation loss, training loop, text generation, decoding strategies and saving/loading weights | Reproducible pretraining and evaluation loop |
| Chapters 6–7 | Classification and instruction fine-tuning material | Read for context if useful, but implementation is outside the current scope |

The book demonstrates its pipeline using a small text sample for learning and debugging. It also discusses larger public-domain book collections as a possible scale-up. A tiny example corpus is useful for proving that the code runs; it is not evidence that a 124M-parameter model has learned broad language competence.

### 2.2 The supplied notebook

The uploaded `LLM Decoding Strategies.ipynb` follows the book's progression and is our **reference implementation**, not yet the final project codebase.

The notebook contains sections for:
- Reading `the-verdict.txt` and splitting text into tokens.
- Constructing a simple word-level vocabulary and tokenizer examples.
- Introducing BPE through the `tiktoken` GPT-2 encoding.
- Creating sliding-window input/target sequences and data loaders.
- Implementing attention components, including causal attention and multi-head attention.
- Building the GPT model from configuration, embeddings, Transformer blocks and an output head.
- Computing language-model loss and training/validation loss.
- Running a simple training loop and producing sample text.
- Decoding strategies including temperature scaling, top-k sampling and multinomial sampling.

The notebook has many pedagogical examples, repeated demonstrations and cells intended to be run in sequence. It should not simply be converted wholesale into a training script. We will extract the concepts and implementation in small pieces, add tests, and make execution order explicit.

### 2.3 Initial notebook audit: important caveats

The notebook's first data-loading example opens `the-verdict.txt` from a local path. That file is not embedded in the notebook itself, so a clean run requires us to supply it or change the dataset loader to download/use a documented corpus.

The notebook's small story is a teaching/debugging corpus. It is not suitable for claiming successful broad pretraining of a 124M model. On such a small dataset, the model may overfit and simply memorize passages.

The notebook contains a basic training loop, but our project still needs a deliberately designed checkpoint format that stores model weights, optimizer state, training step/epoch, configuration and reproducibility metadata. We will also add clean-run tests and explicit validation before any long training run.

A notebook cell having a saved output or execution count does not prove that the entire notebook runs successfully from a clean kernel in the current environment. We must test the extracted code ourselves.

---

## 3. The architecture we intend to implement

Jivi-LLM will be a **decoder-only Transformer**. It predicts the next token from the preceding tokens. During training, a sequence is shifted so that each position is trained to predict the token immediately to its right.

High-level data flow:

```text
Raw text
  │
  ▼
Tokenizer (GPT-2-style BPE for the reference configuration)
  │
  ▼
Token IDs ──► input IDs and shifted target IDs
  │
  ▼
Token embedding + learned positional embedding
  │
  ▼
Dropout
  │
  ▼
Repeated Transformer blocks (12)
  ├── LayerNorm
  ├── Causal multi-head self-attention (12 heads)
  ├── Residual/shortcut connection
  ├── LayerNorm
  ├── Feed-forward network (Linear → GELU → Linear)
  └── Residual/shortcut connection
  │
  ▼
Final LayerNorm
  │
  ▼
Linear language-model head
  │
  ▼
Logits for every vocabulary token at every sequence position
  │
  ├── During training: compare logits with shifted targets using cross-entropy
  │
  └── During generation: sample/select the next token, append it, and repeat
```

### 3.1 Tensor-shape conventions

We will use:
- `B`: batch size
- `T`: sequence/context length
- `C`: embedding dimension
- `V`: vocabulary size
- `H`: number of attention heads
- `D = C / H`: per-head dimension

For the reference configuration:
- Token IDs: `(B, T)`
- Token embeddings: `(B, T, C)`
- Position embeddings: `(T, C)` and broadcast across the batch
- Hidden state through the Transformer: `(B, T, C)`
- Logits: `(B, T, V)`
- Shifted targets: `(B, T)` when using all next-token positions

Every module will have explicit shape checks or tests where useful. Attention must not allow a position to read future tokens.

### 3.2 Attention, conceptually

Given hidden states, learned linear projections produce queries `Q`, keys `K` and values `V`. Attention scores compare queries with keys, scale by the square root of the per-head dimension, apply a causal mask, then use softmax to form weights over allowed earlier/current positions. The weights combine the values. Multiple heads learn different attention patterns, and their outputs are combined through an output projection.

The causal mask is essential: when predicting token `t`, the model must not use the true tokens after `t`. Without this restriction, training would leak the answer.

### 3.3 Transformer block

Each block contains two main sublayers:
1. Causal multi-head self-attention.
2. A position-wise feed-forward network, typically expanding the channel dimension, applying GELU, and projecting back.

Layer normalization and residual/shortcut connections support stable optimization and gradient flow. We will follow the book's block ordering and implementation choices rather than quietly mixing in a different architecture.

### 3.4 Output and loss

The output head maps each hidden vector to `V` logits. Cross-entropy compares those logits with the correct next-token IDs. The model learns by backpropagating the loss and updating its parameters with an optimizer such as AdamW.

A decreasing training loss is necessary evidence that optimization is working, but it is not enough. Validation loss and qualitative samples help detect overfitting and reveal whether the model is learning useful patterns beyond memorizing the training text.

---

## 4. Target model configuration

We will use the book's GPT-2 small-style configuration as the reference target.

| Setting | Reference value | Notes |
|---|---:|---|
| Vocabulary size | 50,257 | Matches the GPT-2 tokenizer vocabulary |
| Context length | 1,024 tokens | Maximum reference sequence length |
| Embedding dimension | 768 | Hidden/channel dimension |
| Transformer blocks | 12 | Number of stacked blocks |
| Attention heads | 12 | 64 dimensions per head (`768 / 12`) |
| Feed-forward hidden dimension | 3,072 | Four times the embedding dimension |
| Parameters | Approximately 124M | Verify exact count from our implementation |

These settings define the architecture; they do not define how much data or compute is sufficient to train it well.

### Memory-aware development

The 124M target is our final architecture target, but we do not need to use every full-scale setting during the first coding steps. On an 8 GB unified-memory Mac, a 124M model can be tight once the operating system, Python, activations, gradients, optimizer states and data are included. Actual feasibility depends on batch size, context length, dtype, PyTorch/MPS behavior and available memory.

We will therefore:
- Develop and test each component on CPU-sized synthetic inputs first.
- Use a smaller temporary configuration for rapid architecture and training-loop tests.
- Run the 124M parameter-count and forward/backward tests on the Mac with a conservative batch size and shorter context first.
- Increase context length and batch size only after observing actual memory use.
- Avoid attempting large-corpus training on the Mac.
- Move compute-heavy experiments to the remote RTX 4090 when the pipeline and dataset justify it.

A small configuration is a test configuration only. We will not accidentally label it as the final Jivi-LLM-124M model.

---

## 5. Tokenizer and corpus strategy

### 5.1 Tokenizer

The notebook first explains tokenization with a simple word-level tokenizer. This is pedagogically helpful because it exposes vocabulary construction, token-to-ID conversion and decoding. However, a vocabulary created from a single story cannot cover arbitrary text and does not match the 50,257-token GPT-2-style configuration.

For the 124M reference model, we will use the GPT-2-style BPE tokenizer exposed by `tiktoken` so that the tokenizer and vocabulary configuration are consistent. Using a tokenizer implementation is not the same as using pretrained model weights; the neural network parameters will still be initialized randomly.

We must preserve tokenizer consistency across dataset preparation, training, checkpoint metadata and generation. Changing tokenization after training begins changes the meaning of token IDs and invalidates the learned embedding/output mappings.

### 5.2 First corpus: debugging, not capability claims

Our first run will use a small, public-domain text corpus, ideally the book's `The Verdict` example or an equivalent source with clear provenance. This makes the first end-to-end experiment manageable and keeps us close to the book's examples.

The first corpus will be used to verify:
- Text can be read and tokenized.
- Input and target sequences align correctly.
- The model produces logits with the expected shape.
- Loss is finite and gradients flow.
- A short training run can reduce loss.
- Checkpoints save and reload.
- Generation runs without shape or device errors.

We should expect the model to overfit a tiny story. It is a pipeline smoke test, not the final pretraining corpus.

### 5.3 Later corpus scale-up

Before using a larger corpus, we will document its source, license/public-domain status, language, size, deduplication strategy, text cleaning, train/validation split and token count. The validation split must be held out from training. If overlapping or near-duplicate text leaks into validation, the reported score can be misleading.

We will estimate the corpus in **tokens**, not only megabytes or number of documents. We will also record the number of training tokens processed and how many times the data is repeated across epochs.

A 124M model trained on a small corpus is still a 124M-parameter model, but that does not mean it is well-trained. Data quantity and diversity matter.

---

## 6. Block-by-block implementation plan

We will build one conceptual unit at a time. Each block must have a clear explanation, implementation, test and checkpoint in our progress before we move to the next.

### Milestone 0 — Project environment and reproducibility

**Purpose:** make the Mac environment predictable before writing model code.

Tasks:
- Create a dedicated Python virtual environment.
- Record Python, PyTorch and `tiktoken` versions.
- Verify CPU execution first and check whether PyTorch MPS is available.
- Establish one configuration file and random seeds.
- Create a clean project structure and a simple test command.
- Keep data, checkpoints and logs separate from source code.

Acceptance criteria:
- Imports work in a fresh environment.
- The chosen device is reported explicitly.
- A small test script runs without relying on notebook cell state.

### Milestone 1 — Text and token IDs

**Purpose:** understand the exact representation the model receives.

Tasks:
- Read a documented public-domain text file.
- Inspect its character count and a small excerpt.
- Encode and decode with the selected tokenizer.
- Check that decoded text is consistent with the tokenizer's expected behavior.
- Record token count and dataset provenance.

Acceptance criteria:
- Tokenization is deterministic for the same input.
- IDs are integers in `[0, vocab_size)`.
- The configured vocabulary size agrees with the tokenizer.
- Missing corpus files produce a helpful error rather than a confusing traceback.

### Milestone 2 — Sliding-window training examples

**Purpose:** teach the model next-token prediction.

For a token sequence such as:

```text
Tokens:   [A, B, C, D, E]
Input:    [A, B, C, D]
Target:   [B, C, D, E]
```

The target is shifted one token to the right. At each position, the model is trained to predict the next token. A sliding window creates many examples from a longer text sequence.

Tasks:
- Implement the dataset/window logic.
- Split training and validation data without leakage.
- Create batches and inspect shapes.
- Test stride, context length, short text and boundary cases.

Acceptance criteria:
- Inputs and targets have the expected dimensions.
- Every target sequence is shifted correctly.
- No example uses tokens from the held-out validation region in its training input.

### Milestone 3 — Embeddings

**Purpose:** map discrete token IDs to vectors and encode position.

Tasks:
- Implement token embeddings.
- Implement learned positional embeddings.
- Add them and check resulting shapes.
- Test valid token and position ranges.

Acceptance criteria:
- Output shape is `(B, T, C)`.
- Invalid token IDs or positions fail clearly.
- Embedding parameters receive gradients.

### Milestone 4 — Self-attention

**Purpose:** implement how each token gathers information from allowed earlier tokens.

Tasks:
- Implement query, key and value projections.
- Calculate scaled attention scores.
- Apply the causal mask before softmax.
- Add attention dropout as prescribed by the chosen book configuration.
- Combine values using attention weights.
- Test shapes, finite values and causal behavior.

Acceptance criteria:
- A position cannot attend to future positions.
- Output dimensions are correct.
- Forward and backward passes work on a small tensor.

### Milestone 5 — Multi-head attention

**Purpose:** allow multiple attention heads to learn different relationships.

Tasks:
- Split channels into heads.
- Compute attention efficiently with batched tensor operations.
- Concatenate head outputs and apply the output projection.
- Verify dropout and dimensions.

Acceptance criteria:
- Output shape matches the input hidden-state shape.
- Head reshaping is reversible and correct.
- Gradients reach all projection parameters.

### Milestone 6 — Feed-forward network and normalization

**Purpose:** transform token representations after attention.

Tasks:
- Implement LayerNorm.
- Implement the feed-forward network using Linear → GELU → Linear.
- Check the expected expansion and contraction of dimensions.
- Test finite output and gradients.

Acceptance criteria:
- Feed-forward output returns to the embedding dimension.
- LayerNorm operates over the intended channel dimension.
- No NaNs or infinities appear in normal test cases.

### Milestone 7 — Residual paths and complete Transformer block

**Purpose:** connect the attention and feed-forward sublayers while preserving information and supporting gradient flow.

Tasks:
- Implement the book's normalization/residual ordering.
- Add the appropriate residual/shortcut connections and dropout.
- Test a block independently before stacking it.

Acceptance criteria:
- Input and output shapes match.
- The block supports a backward pass.
- Residual paths are not accidentally detached from autograd.

### Milestone 8 — GPT model and output head

**Purpose:** assemble the complete decoder-only architecture.

Tasks:
- Stack 12 Transformer blocks for the final configuration.
- Add final LayerNorm and language-model output projection.
- Run a forward pass on a small batch.
- Count parameters and compare with the approximately 124M target.
- Inspect output logits and verify the vocabulary dimension.

Acceptance criteria:
- The full model produces `(B, T, V)` logits.
- The exact parameter count is recorded.
- A forward and backward pass completes under the selected test configuration.
- No pretrained GPT-2 model weights are loaded.

### Milestone 9 — Loss and optimization

**Purpose:** measure prediction error and update parameters.

Tasks:
- Align model logits and target IDs.
- Compute cross-entropy over all intended token positions.
- Initialize AdamW with explicit hyperparameters.
- Zero gradients, backpropagate and step the optimizer.
- Check gradients and parameter updates on a tiny controlled batch.

Acceptance criteria:
- Loss is finite.
- Gradients are nonzero for expected trainable parameters.
- A simple repeated-batch overfit test can reduce loss, confirming that learning works.

### Milestone 10 — Evaluation and pretraining loop

**Purpose:** make training measurable and reproducible.

Tasks:
- Implement training and validation loss evaluation.
- Switch model train/eval modes correctly.
- Disable gradient tracking during evaluation.
- Log step, tokens seen, train loss, validation loss, learning rate and elapsed time.
- Generate fixed-prompt samples at evaluation intervals.
- Add early-stop/step limits for experiments and handle interruption safely.

Acceptance criteria:
- Metrics are recorded in a machine-readable log.
- Validation does not update weights.
- The same configuration and seed are recorded with each run.
- A short run completes without unbounded memory growth.

### Milestone 11 — Checkpoint and resume

**Purpose:** avoid losing expensive training progress.

Each checkpoint should include:
- Model state dictionary.
- Optimizer state dictionary.
- Step and epoch.
- Configuration.
- Random-number generator state where practical.
- Tokens seen and recent training/validation metrics.
- Tokenizer identifier/version and dataset metadata.

Acceptance criteria:
- Save a checkpoint, load it into a fresh model/optimizer, and continue training.
- Verify loaded model outputs match the saved model for the same input in evaluation mode.
- A resumed run records that it was resumed and from which checkpoint.

### Milestone 12 — Decoding and generation evaluation

**Purpose:** inspect what the pretrained base model has learned.

We will use the notebook's decoding concepts:
- Greedy selection / argmax as a baseline.
- Temperature scaling.
- Top-k sampling.
- Multinomial sampling from the filtered probability distribution.
- Context-window truncation for inputs longer than the configured maximum.

We will check that temperature is positive when used as a divisor, top-k is within the vocabulary range, and sampling probabilities are finite and valid. EOS stopping is relevant only if the chosen tokenizer/data convention and generation function define and use an EOS token correctly.

Acceptance criteria:
- Generation returns valid token IDs and decodes to text.
- Generation respects the context limit.
- The same prompt can be tested with different decoding settings.
- Generated samples are saved alongside the model step and configuration.

---

## 7. Proposed project layout

We will create and populate these files incrementally, not all at once:

```text
jivi-llm/
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml              # or requirements.txt, chosen at setup
├── configs/
│   ├── debug.yaml
│   └── gpt_124m.yaml
├── data/
│   ├── raw/                    # source text; normally not committed
│   └── processed/              # tokenized/cache files; normally not committed
├── checkpoints/                # ignored by Git
├── logs/
├── src/
│   └── jivi_llm/
│       ├── __init__.py
│       ├── config.py
│       ├── tokenizer.py
│       ├── dataset.py
│       ├── attention.py
│       ├── transformer_block.py
│       ├── model.py
│       ├── loss.py
│       ├── train.py
│       ├── evaluate.py
│       ├── checkpoint.py
│       └── generate.py
└── tests/
    ├── test_tokenizer.py
    ├── test_dataset.py
    ├── test_attention.py
    ├── test_model.py
    └── test_checkpoint.py
```

We may adjust module boundaries as the implementation becomes clearer. The important principles are separation of concerns, reproducible configuration and tests.

Large data files, caches, checkpoints, and logs should not be committed to Git by default. They can be stored separately and described through metadata/manifests.

---

## 8. Hardware and training strategy

### On the Mac first

The Mac is the development and verification machine. We can learn the architecture, run unit tests, check tensor shapes, execute short CPU/MPS smoke tests, and train on a tiny corpus to validate the entire loop.

We should not assume that the M3's integrated GPU behaves identically to CUDA. Device-specific operations, dtype support, memory limits and performance need to be tested. Every experiment should report its actual device rather than silently switching hardware.

Initial experiments should prioritize correctness over speed:
- Short sequence length.
- Small batch size.
- A debug-sized model where necessary.
- Few optimizer steps.
- Frequent loss and memory checks.

### Later on the remote RTX 4090

Move to the remote GPU after:
1. The architecture and training loop pass tests.
2. Checkpoint/resume works.
3. Dataset preparation is documented and repeatable.
4. A short benchmark measures tokens/second and memory use.
5. We know what experiment we intend to run and how success will be evaluated.

The GPU machine will be used for compute-heavy training, while code can still be developed and reviewed locally. Before scaling up, we will verify CUDA/PyTorch compatibility, disk capacity, available VRAM and remote checkpoint backup.

A 24 GB GPU does not guarantee that every batch/context/dtype configuration fits. We will benchmark rather than guess.

---

## 9. What success means

We will distinguish software correctness, optimization progress and language-model quality.

### Level A — Component correctness
- All unit tests pass.
- Causal masking is verified.
- Tensor shapes and parameter count match the configuration.
- Gradients flow as expected.

### Level B — End-to-end pipeline correctness
- The model trains from random initialization.
- Loss is finite and can decrease on a controlled tiny dataset.
- Validation is separate from training.
- Checkpoints save, reload and resume.
- Generation works after reloading.

### Level C — Corpus experiment
- Dataset provenance and token count are documented.
- Training/validation losses are plotted.
- Tokens processed, elapsed time, configuration and compute device are recorded.
- Sample generations are compared across checkpoints.
- Overfitting and limitations are discussed honestly.

### Level D — Larger-scale pretraining
- A larger, legally usable and appropriately curated corpus is prepared.
- Data processing and storage are repeatable.
- Training duration and token budget are selected based on a measured GPU benchmark.
- Evaluation is tracked over time.
- Results are reported with enough detail to reproduce the experiment.

A falling training loss alone is not proof of general language ability. If the tiny corpus is memorized, that is a useful debugging result but not a successful general-purpose LLM.

---

## 10. Engineering rules for this project

1. **Build block by block.** Do not dump a complete codebase before we understand and test its parts.
2. **Explain before implementing.** For each block, describe the purpose, equations, tensor shapes and common failure modes.
3. **Test each block.** Write a small test or numerical check before integrating the next component.
4. **Use random initialization.** No pretrained GPT-2 weights or other pretrained language-model weights.
5. **Keep the tokenizer fixed.** Save tokenizer identity and vocabulary configuration with every checkpoint.
6. **Separate debug and final configurations.** Never confuse a small smoke-test model with the final 124M model.
7. **Measure rather than assume.** Record device, memory, tokens/second, loss and training duration.
8. **Make runs resumable.** Save optimizer state and progress, not just model weights.
9. **Protect validation data.** Do not use validation examples for gradient updates.
10. **Be honest about quality.** A randomly initialized GPT trained on a small text file will not become a capable assistant simply because the script finishes.
11. **No unnecessary production features.** We are not building serving infrastructure or instruction fine-tuning in this phase.
12. **Keep experiments reproducible.** Version configuration, document data sources, seed randomness where possible, and record dependencies.

---

## 11. Immediate next steps

The first implementation session should be deliberately small.

### Step 1 — Inspect and establish the environment
- Confirm the project directory and Python version.
- Create a virtual environment.
- Install only the dependencies needed for the first blocks.
- Check PyTorch, CPU and MPS availability.

### Step 2 — Revisit the notebook's data/tokenizer section
- Locate or acquire the public-domain `The Verdict` text.
- Reproduce the notebook's tokenization examples.
- Confirm the selected GPT-2-style BPE tokenizer and vocabulary size.
- Write tokenizer tests.

### Step 3 — Build the sliding-window dataset
- Inspect a few token IDs and decoded samples.
- Verify shifted inputs and targets manually.
- Build a tiny batch and test all shapes.

### Step 4 — Begin model components
- Implement token/position embeddings.
- Then implement single-head causal attention and test the causal mask.
- Only after this passes should we move to multi-head attention.

**Do not start a long training run yet.** We should first prove that the data, shapes, mask and gradients are correct.

---

## 12. Final definition of Jivi-LLM v0

For the first complete milestone, Jivi-LLM v0 means:

> A GPT-2-style, approximately 124M-parameter decoder-only Transformer whose weights are initialized from scratch, whose training examples come from a documented public-domain text corpus, and which can be pretrained using next-token prediction, evaluated with held-out loss, saved/resumed from checkpoints, and used to generate text for research and learning.

It does **not** mean a pretrained OpenAI GPT model, an instruction-following assistant, or a production-ready product.

The guiding principle is simple: **understand it, implement it, test it, then scale it.**
