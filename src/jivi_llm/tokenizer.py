import tiktoken

def get_tokenizer(name="gpt2"):
    tokenizer = tiktoken.get_encoding(name)
    return tokenizer

def encode(text, tokenizer): 
    # string to int ids
    return tokenizer.encode(text)

def decode(text, tokenizer):
    # ids to string
    return tokenizer.decode(text)

    