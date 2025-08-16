import os
from typing import BinaryIO
import regex as re
import multiprocessing

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

# Single thread of pretokenization
# read chunk and return all possible words and count.
def single_pretokenize(chunk : str, special_tokens: list[str], shared_dict : dict[str, int], lock):
    result = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    strings = re.split("|".join(special_tokens), chunk) 
            
    for string in strings:
        keys = re.finditer(PAT, string)
        
        for match in keys:
            key = match.group()
            if key in result:
                result[key] += 1
            else:
                result[key] = 1
            
    with lock:
        for key, value in result.items():
            if key in shared_dict:
                shared_dict[key] += value
            else:
                shared_dict[key] = value
    return

# Multithread
def pretokenize(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int) -> dict[str, int]:
    manager = multiprocessing.Manager()
    shared_dict = manager.dict({})
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        processes = []
        # print(boundaries)
        lock = multiprocessing.Lock()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            process = multiprocessing.Process(target = single_pretokenize, args = (chunk, special_tokens, shared_dict, lock))
            processes.append(process)
            process.start()

        for p in processes:
            p.join()

    result = dict(shared_dict)
    return result

# Singlethread
def pretokenize(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[str, int]:
    result = {}
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(input_path, "rb") as f:
        chunk = f.read().decode("utf-8", errors="ignore")
        strings = re.split("|".join(special_tokens), chunk) 
            
    for string in strings:
        keys = re.finditer(PAT, string)
        
        for match in keys:
            key = match.group()
            if key in result:
                result[key] += 1
            else:
                result[key] = 1
    return result

def translator(word: str) -> list[int]:
    return list(word.encode("utf-8"))

def words_map_initializer(words: dict[str, int]) -> dict[str, tuple[int, list[int]]]:
    res = {}
    for word in words:
        res[word] = (words[word], translator(word))
    return res

def vocab_initializer(special_tokens: list[str]) -> dict[int, bytes]:
    res = {}
    for i in range(256):
        res[i] = bytes([i])
    l = len(special_tokens)
    for i in range(l):
        res[i + 256] = bytes(special_tokens[i], "utf-8")
    return res

def pair_updater(old: tuple[int, list[str]], val: int, words: list[str]) -> tuple[int, list[str]]:
    if val > 0:
        res = (old[0] + val, old[1] + words)
    else:
        assert words[0] in old[1], f"words = {words}, old[1] = {old[1]}"
        assert old[0] + val >= 0, f"words = {words}, old[1] = {old[1]}"
        for i in words:
            old[1].remove(i)
        res = (old[0] + val, old[1])
    return res

def counter(words_map: dict[str, tuple[int, list[int]]]) -> dict[tuple[int, int], tuple[int, list[str]]]:
    res = {}
    words = words_map.keys()
    for word in words:
        word_list = words_map[word][1]
        for left, right in zip(word_list[:-1], word_list[1:]):
            key = (left, right)
            if key in res:
                res[key] = pair_updater(res[key], words_map[word][0], [word])
            else:
                res[key] = (words_map[word][0], [word])
    return res
    
def selector(tokens_map: dict[tuple[int, int], tuple[int, list[str]]], vocab: dict[int, bytes]) -> tuple[int, int]:
    return max(tokens_map, key=lambda x: (tokens_map[x][0], vocab[x[0]], vocab[x[1]]))

def update_words_map(val: int, l: list[int], choice: tuple[int, int], new_num: int) -> tuple[tuple[int, list[int]], list[int]]:
    new_list = []
    loc = []
    jump = False
    bound = len(l) - 1
    for i in range(bound):
        # Update words_map
        if jump:
            jump = False
        else:
            if (l[i], l[i + 1]) == choice:
                new_list.append(new_num)
                loc.append(i)
                jump = True
            else:
                new_list.append(l[i])
    # insert last character if needed
    if not jump:
        new_list.append(l[-1])
    return ((val, new_list), loc)

def update_tokens_pairs(loc: list[int], word: str, val: int, l: list[int], tokens_map: dict[tuple[int, int], tuple[int, list[str]]]):
    pairs = []
    bound = len(l)
    for iter in loc:
        pairs.append((iter - 1, iter))
        pairs.append((iter, iter + 1))
        pairs.append((iter + 1, iter + 2))
    for pair in set(pairs):
        if pair[0] >= 0 and pair[1] < bound:
            key = (l[pair[0]], l[pair[1]])
            tokens_map[key] = pair_updater(tokens_map[key], -val, [word])
    return

def add_tokens_pairs(word: str, val: int, l: list[int], tokens_map: dict[tuple[int, int], tuple[int, list[str]]], new_num: int):
    for left, right in zip(l[:-1], l[1:]):
        if left == new_num or right == new_num:
            key = (left, right)
            if key in tokens_map:
                tokens_map[key] = pair_updater(tokens_map[key], val, [word])
            else:
                tokens_map[key] = (val, [word])
    return

def merge(words_map: dict[str, tuple[int, list[int]]], tokens_map: dict[tuple[int, int], tuple[int, list[str]]], choice: tuple[int, int], new_num: int) -> tuple[dict[str, tuple[int, list[int]]], dict[tuple[int, int], tuple[int, list[str]]]]:
    words = tokens_map[choice][1]
    # words can occur multiple times.
    for word in set(words):
        # Update words_map
        res = update_words_map(words_map[word][0], words_map[word][1], choice, new_num)
        # Update tokens_map(Del)
        update_tokens_pairs(res[1], word, words_map[word][0], words_map[word][1], tokens_map)
        words_map[word] = res[0]
        # Update tokens_map(Add)
        add_tokens_pairs(word, words_map[word][0], words_map[word][1], tokens_map, new_num)
    tokens_map[choice] = (0, [])
    return words_map, tokens_map

"""Given the path to an input corpus, run train a BPE tokenizer and
output its vocabulary and merges.

Args:
    input_path (str | os.PathLike): Path to BPE tokenizer training data.
    vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
    special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
        These strings will never be split into multiple tokens, and will always be
        kept as a single token. If these special tokens occur in the `input_path`,
        they are treated as any other string.

Returns:
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab:
            The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges:
            BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
"""
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Maybe because less data, single thread is faster
    if 'num_processes' in kwargs:
        words_cnt = pretokenize(input_path, special_tokens, kwargs['num_processes'])
    else:
        words_cnt = pretokenize(input_path, special_tokens)
    vocab = vocab_initializer(special_tokens)
    merges = []
    words_map = words_map_initializer(words_cnt)
    begin = len(vocab)
    tokens_map = counter(words_map)
    for i in range(begin, vocab_size):
        choice = selector(tokens_map, vocab)
        # (32, 115) means (b' ', b's'), (101, 114) means (b'e', b'r')
        # print("(b' ', b's') = ", tokens_map[(32, 115)][0], "(b'e', b'r') = ", tokens_map[(101, 114)][0])
        # print(vocab[choice[0]], vocab[choice[1]], tokens_map[choice][0], len(tokens_map[choice][1]))
        words_map, tokens_map = merge(words_map, tokens_map, choice, i)
        # print(type(words_map), type(tokens_map))
        choice = (vocab[choice[0]], vocab[choice[1]])
        merges.append(choice)
        vocab[i] = choice[0] + choice[1]
    return (vocab, merges)