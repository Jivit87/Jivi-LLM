import os

import tiktoken


class CustomTokenizer:
    """Wraps a Hugging Face `tokenizers` BPE file (e.g. the medical 32k BPE) with the same encode/decode interface as tiktoken."""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self._tok = Tokenizer.from_file(path)
        self.n_vocab = self._tok.get_vocab_size()
        self.eot_token = self._tok.token_to_id("<|endoftext|>")

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(list(ids))


def get_tokenizer(name="gpt2"):
    # a path to a tokenizer.json file selects the custom tokenizer; anything else is a tiktoken encoding name
    if name.endswith(".json") or os.path.isfile(name):
        return CustomTokenizer(name)
    tokenizer = tiktoken.get_encoding(name)
    return tokenizer

def encode(text, tokenizer): 
    # string to int ids
    return tokenizer.encode(text)

def decode(text, tokenizer):
    # ids to string
    return tokenizer.decode(text)
