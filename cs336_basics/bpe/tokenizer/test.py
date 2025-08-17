# uv run cs336_basics/bpe/tokenizer/test.py
print(ord("🙃"))
l = list("🙃 123".encode())
print(l)
print(bytes(l).decode(errors='replace'))
print(2**8)

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

test_string = "Hello, how are you?"
ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
print(ids[4])
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
test_string = "Héllò hôw are ü? 🙃"
ids = tokenizer.encode(test_string, allowed_special={"<|endoftext|>"})
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)

from tokenizer import Tokenizer
import pathlib
FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent)
vocab_file = FIXTURES_PATH / ".." / "train" / "vocab.txt"
merges_file = FIXTURES_PATH / ".." / "train" / "merges.txt"
tokenizer = Tokenizer.from_files(Tokenizer, vocab_file, merges_file, ["<|endoftext|>"])

test_string = "s"
ids = tokenizer.encode(test_string)
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
test_string = "\n\n\n\n"
ids = tokenizer.encode(test_string)
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
test_string = "Hello, how are you?"
ids = tokenizer.encode(test_string)
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
ids = tokenizer.encode(test_string)
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
test_string = "Héllò hôw are ü? 🙃"
ids = tokenizer.encode(test_string)
tokenized_string = [tokenizer.decode([x]) for x in ids]
print(tokenized_string)
