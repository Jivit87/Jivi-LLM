import pytest
import sys
sys.path.insert(0, "src")

from jivi_llm.dataset import GPTDataset, create_dataloader


# simple repeating token sequence for predictable testing
TOKENS = list(range(50))


def test_target_is_input_shifted_by_one():
    # target[i] must equal input[i+1] for every position
    ds = GPTDataset(TOKENS, context_length=8, stride=1)
    inp, tgt = ds[0]
    assert inp[1:].tolist() == tgt[:-1].tolist()


def test_input_target_shape():
    # each sample must be a 1D tensor of length context_length
    context_length = 8
    ds = GPTDataset(TOKENS, context_length=context_length, stride=1)
    inp, tgt = ds[0]
    assert inp.shape == (context_length,)
    assert tgt.shape == (context_length,)


def test_stride_reduces_dataset_size():
    # larger stride means fewer windows
    ds_stride1 = GPTDataset(TOKENS, context_length=8, stride=1)
    ds_stride4 = GPTDataset(TOKENS, context_length=8, stride=4)
    assert len(ds_stride4) < len(ds_stride1)


def test_dataloader_batch_shape():
    # batches from the dataloader must have correct shape
    dl = create_dataloader(TOKENS, context_length=8, stride=1, batch_size=4, shuffle=False)
    inp_batch, tgt_batch = next(iter(dl))
    assert inp_batch.shape == (4, 8)
    assert tgt_batch.shape == (4, 8)
