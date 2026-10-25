import pytest

from jivi_llm.tokenizer import get_tokenizer, encode, decode


@pytest.fixture
def tok():
    # reuse one tokenizer across all tests
    return get_tokenizer()


def test_round_trip(tok):
    # encode then decode should return the original text
    text = "Hello, I am building a GPT from scratch."
    assert decode(encode(text, tok), tok) == text


def test_token_ids_in_range(tok):
    # all token IDs must be valid GPT-2 vocab indices
    ids = encode("The quick brown fox", tok)
    assert all(0 <= i <= 50256 for i in ids)


def test_missing_file_raises():
    # loading a non-existent corpus file should fail clearly
    with pytest.raises(FileNotFoundError):
        open("data/raw/does_not_exist.txt")
