from typing import Any, Iterable, Iterator
import regex as re
import os

# uv run pytest tests/test_tokenizer.py
def pretokenize(inp: str, special_tokens: list[str] | None) -> list[str]:
    result = []
    if inp == '':
        return result
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens is None:
        strings = [inp]
    else:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_tokens = [f"({re.escape(token)})" for token in sorted_tokens]
        pattern = "|".join(escaped_tokens)
        strings = re.split(pattern, inp)

    for string in strings:
        if string is None or string == '':
            continue
        elif special_tokens is not None and string in special_tokens:
            result.append(string)
            continue

        keys = re.finditer(PAT, string)
        for match in keys:
            word = match.group()
            if word is None or word == '':
                continue
            result.append(word)
    return result

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return Tokenizer(vocab, merges, special_tokens)
    # raise NotImplementedError

class Tokenizer:
    """
    Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        self.byte_map = {v: k for k, v in vocab.items()}
        if special_tokens is not None:
            for token in special_tokens:
                b = token.encode()
                if b not in self.byte_map:
                    bound = len(self.byte_map)
                    self.byte_map[b] = bound
                    self.vocab[bound] = b

        # Using cache here is way more faster.
        self.cache: dict[str, list[int]] = {}
        self.pair_map: list[tuple[tuple[int, int], int]] = []
        for merge in merges:
            left = self.byte_map[merge[0]]
            right = self.byte_map[merge[1]]
            key = (left, right)
            self.pair_map.append((key, self.byte_map[merge[0] + merge[1]]))
        return
    
    """
    Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges(in the same format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
    """
    def from_files(cls, vocab_filepath: str | os.PathLike, merges_filepath: str | os.PathLike, special_tokens: list[str] | None=None):
        vocab = {}
        with open(vocab_filepath) as f:
            while True:
                string = f.readline()
                if string == "":
                    break
                matches = re.findall(r"(\d+) (b'.*?'$)", string)
                if matches == []:
                    matches = re.findall(r'(\d+) (b".*?"$)', string)
                assert len(matches) == 1, f"matches = {matches}, string = {string}"
                res = matches[0]
                vocab[res[0]] = eval(res[1])
        f.close()
        
        merges = []
        with open(merges_filepath) as f:
            while True:
                string = f.readline()
                if string == "":
                    break
                matches = re.findall(r"(b'.*?') (b'.*?'$)", string)
                if matches == []:
                    matches = re.findall(r"(b'.*?') (b\".*?\"$)", string)
                if matches == []:
                    matches = re.findall(r'(b".*?") (b\'.*?\'$)', string)
                if matches == []:
                    matches = re.findall(r'(b".*?") (b".*?"$)', string)
                assert len(matches) == 1, f"matches = {matches}, string = {string}"
                res = matches[0]
                merges.append((eval(res[0]), eval(res[1])))
        f.close()
        return Tokenizer(vocab, merges, special_tokens)
    
    def translator(self, word: str) -> list[int]:
        l = word.encode("utf-8")
        res = []
        for i in range(len(l)):
            assert self.byte_map[l[i: i+1]] is not None
            res.append(self.byte_map[l[i: i+1]])
        return res

    def merge(self, l: list[int]) -> list[int]:
        if l is None or l == []:
            return []
        elif len(l) == 1:
            return l
        
        pairs = list(zip(l[:-1], l[1:]))
        for merge in self.pair_map:
            if merge[0] not in pairs:
                continue
            new = []
            jump = False
            for pair in pairs:
                if jump:
                    jump = False
                elif merge[0] == pair:
                    new.append(merge[1])
                    jump = True
                else:
                    new.append(pair[0])
            if not jump:
                new.append(l[-1])
            l = new
            if len(l) == 1:
                break
            pairs = list(zip(l[:-1], l[1:]))           
        return l
    
    """
    Encode an input text into a sequence of token IDs.
    """
    def encode(self, text: str) -> list[int]:
        words = pretokenize(text, self.special_tokens)
        # print(words)
        
        res = []
        for word in words:
            if self.special_tokens is not None and word in self.special_tokens:
                res.append(self.byte_map[word.encode()])
            elif word in self.cache:
                res += self.cache[word]
            else:
                l = self.translator(word)
                # print(self.merge(l))
                ans = self.merge(l)
                self.cache[word] = ans
                res += ans
        return res
    
    """
    Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
    """
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            res = self.encode(string)
            for i in res:
                yield i
    
    """
    Decode a sequence of token IDs into text.
    """
    def decode(self, ids: list[int]) -> str:
        res = b''
        for id in ids:
            res += self.vocab[id]
        return res.decode(errors='replace')