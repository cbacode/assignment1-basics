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

def word_initializer(words: dict[str, int]) -> dict[str, list[int]]:
    res = {}
    for word in words:
        res[word] = translator(word)
    return res

def vocab_initializer() -> dict[int, bytes]:
    res = {}
    for i in range(256):
        res[i] = bytes([i])
    return res

def single_counter(words: list[str], words_cnt: dict[str, int], words_list: dict[str, list[int]], shared_dict: dict[tuple[int, int], int], lock):
    res = {}
    for word in words:
        word_list = words_list[word]
        for left, right in zip(word_list[:-1], word_list[1:]):
            if (left, right) in res:
                res[(left, right)] += words_cnt[word]
            else:
                res[(left, right)] = words_cnt[word]
    with lock:
        for key, value in res.items():
            if key in shared_dict:
                shared_dict[key] += value
            else:
                shared_dict[key] = value
    return

def counter(words: list[str], words_cnt: dict[str, int], words_list: dict[str, list[int]], num_process = 8) -> dict[tuple[int, int], int]:
    manager = multiprocessing.Manager()
    shared_dict = manager.dict({})
    processes = []
    words_size = len(words)
    desired_num_chunks = words_size // num_process
    boundaries = [i * desired_num_chunks for i in range(desired_num_chunks + 1)]
    boundaries[-1] = words_size
    # print(boundaries)
    lock = multiprocessing.Lock()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process = multiprocessing.Process(target = single_counter, args = (words[start:end], words_cnt, words_list, shared_dict, lock))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()

    result = dict(shared_dict)
    return result

def selector(cnt: dict[tuple[int, int], int]) -> tuple[int, int]:
    return max(cnt, key=lambda x: (cnt[x], x))

def single_merger(words: list[str], words_list: dict[str, list[int]], choice: tuple[int, int], new_num: int):
    # No need to lock anything
    change = False
    new_list = []
    for word in words:
        l = words_list[word]
        i = 0
        for i in range(len(l) - 1):
            if l[i] == choice[0] and l[i + 1] == choice[1]:
                change = True
                new_list.append(new_num)
                i += 1
            else:
                new_list.append(l[i])
        # insert last character if needed
        if i == len(l) - 1:
            new_list.append(l[i])
        if change :
            words_list[word] = new_list
    return

def merge(words: list[str], words_list: dict[str, list[int]], choice: tuple[int, int], new_num: int, num_process = 8):
    processes = []
    words_size = len(words)
    desired_num_chunks = words_size // num_process
    boundaries = [i * desired_num_chunks for i in range(desired_num_chunks + 1)]
    boundaries[-1] = words_size
    # print(boundaries)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        process = multiprocessing.Process(target = single_merger, args = (words[start:end], words_list, choice, new_num))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()
        
    return

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
    num_processes = 8
    words_cnt = pretokenize(input_path, special_tokens, num_processes)
    words = list(words_cnt.keys())
    vocab = vocab_initializer()
    merges = []
    words_list = word_initializer(words_cnt)
    begin = len(vocab)
    for i in range(begin, vocab_size):
        # print(i)
        cnt = counter(words, words_cnt, words_list, num_processes)
        choice = selector(cnt)
        merge(words, words_list, choice, i, num_processes)
        choice = (vocab[choice[0]], vocab[choice[1]])
        merges.append(choice)
        vocab[i] = choice[0] + choice[1]
    return (vocab, merges)