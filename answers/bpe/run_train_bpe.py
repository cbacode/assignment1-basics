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

def pretokenize(input_path: str | os.PathLike, special_tokens: list[str], num_processes = 8) -> dict[str, int]:
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

def translator(word: str) -> list[int]:
    return list(word.encode("utf-8"))

def words_map_initializer(words: dict[str, int]) -> dict[str, tuple[int, list[int]]]:
    res = {}
    for word in words:
        res[word] = (words[word], translator(word))
    return res

def vocab_initializer() -> dict[int, bytes]:
    res = {}
    for i in range(256):
        res[i] = bytes([i])
    return res

def pair_updater(old: tuple[int, list[str]], val: int, words: list[str]) -> tuple[int, list[str]]:
    if len(words) == 1 and old[1][-1] == words[0]:
        # Words should be single
        res = (old[0] + val, old[1])
    else:
        res = (old[0] + val, old[1] + words)
    return res

def single_counter(words: list[str], words_map: dict[str, tuple[int, list[int]]], shared_dict: dict[tuple[int, int], tuple[int, list[str]]], lock):
    res = {}
    for word in words:
        word_list = words_map[word][1]
        for left, right in zip(word_list[:-1], word_list[1:]):
            key = (left, right)
            if key in res:
                res[key] = pair_updater(res[key], words_map[word][0], [word])
            else:
                res[key] = (words_map[word][0], [word])
    with lock:
        for key, value in res.items():
            if key in shared_dict:
                shared_dict[key] = pair_updater(shared_dict[key], res[key][0], res[key][1])
            else:
                shared_dict[key] = value
    return

def counter(words: list[str], words_map: dict[str, tuple[int, list[int]]], num_process: int = 8) -> dict[tuple[int, int], tuple[int, list[str]]]:
    manager = multiprocessing.Manager()
    shared_dict = manager.dict({})
    processes = []
    words_size = len(words)
    chunk_size = words_size // num_process
    boundaries = [i * chunk_size for i in range(num_process + 1)]
    boundaries[-1] = words_size
    # print(boundaries)
    lock = multiprocessing.Lock()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process = multiprocessing.Process(target = single_counter, args = (words[start:end], words_map, shared_dict, lock))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()

    result = dict(shared_dict)
    return result

def selector(tokens_map: dict[tuple[int, int], tuple[int, list[str]]], vocab: dict[int, bytes]) -> tuple[int, int]:
    return max(tokens_map, key=lambda x: (tokens_map[x][0], vocab[x[0]], vocab[x[1]]))

def single_merge(words: list[str], words_map: dict[str, tuple[int, list[int]]], tokens_map: dict[tuple[int, int], tuple[int, list[str]]], choice: tuple[int, int], new_num: int, lock):
    new_pairs = {}
    for word in words:
        new_list = []
        l = words_map[word][1]
        jump = False
        for left, right in zip(l[:-1], l[1:]):
            # Update words_map(Delete)
            if jump:
                jump = False
            else:
                if (left, right) == choice:
                    new_list.append(new_num)
                    jump = True
                else:
                    new_list.append(left)
            # Update tokens_map      
            if left == choice[1] or right == choice[0]:
                tokens_map[(left, right)] = (0, [])
        # insert last character if needed
        if not jump:
            new_list.append(l[-1])
        words_map[word] = (words_map[word][0], new_list)
        
        # Update tokens_map(Add)
        l = new_list
        for left, right in zip(l[:-1], l[1:]):
            if left == new_num or right == new_num:
                key = (left, right)
                if key in new_pairs:
                    new_pairs[key] = pair_updater(new_pairs[key], words_map[word][0], [word])
                else:
                    new_pairs[key] = (words_map[word][0], [word])
    with lock:
        for key, value in new_pairs.items():
            if key in tokens_map:
                tokens_map[key] = pair_updater(tokens_map[key], new_pairs[key][0], new_pairs[key][1])
            else:
                tokens_map[key] = value
    return

def merge(words_map: dict[str, tuple[int, list[int]]], tokens_map: dict[tuple[int, int], tuple[int, list[str]]], choice: tuple[int, int], new_num: int, num_process: int = 8) -> tuple[dict[str, tuple[int, list[int]]], dict[tuple[int, int], tuple[int, list[str]]]]:
    manager = multiprocessing.Manager()
    shared_tokens_map = manager.dict(tokens_map)
    shared_words_map = manager.dict(words_map)
    processes = []
    words = tokens_map[choice][1]
    words_size = len(words)
    if words_size < num_process:
        boundaries = [i for i in range(words_size + 1)]
    else:
        chunk_size = words_size // num_process
        boundaries = [i * chunk_size for i in range(num_process + 1)]
    boundaries[-1] = words_size
    # print(boundaries)
    lock = multiprocessing.Lock()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process = multiprocessing.Process(target = single_merge, args = (words[start:end], shared_words_map, shared_tokens_map, choice, new_num, lock))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()
    
    words_map = dict(shared_words_map)
    tokens_map = dict(shared_tokens_map)
    tokens_map[choice] = (0, [])
    return (words_map, tokens_map)

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
    num_processes = 32
    words_cnt = pretokenize(input_path, special_tokens, num_processes)
    words = list(words_cnt.keys())
    vocab = vocab_initializer()
    merges = []
    words_map = words_map_initializer(words_cnt)
    begin = len(vocab)
    tokens_map = counter(words, words_map, num_processes)
    for i in range(begin, vocab_size):
        choice = selector(tokens_map, vocab)
        print(vocab[choice[0]], vocab[choice[1]], tokens_map[choice])
        words_map, tokens_map = merge(words_map, tokens_map, choice, i)
        # print(type(words_map), type(tokens_map))
        choice = (vocab[choice[0]], vocab[choice[1]])
        merges.append(choice)
        vocab[i] = choice[0] + choice[1]
    return (vocab, merges)