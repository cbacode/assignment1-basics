import cs336_basics.bpe.train.run_train_bpe as run_train_bpe
import cProfile
import pstats
import pathlib
import sys

# uv run cs336_basics/bpe/train/stories.py

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent)
# FIXTURES_PATH = '/home/cbacoding/llm/CS336/assignment1-basics/cs336_basics/bpe'
input_path = FIXTURES_PATH / '..' / '..' / '..' / 'data' / 'TinyStoriesV2-GPT4-valid.txt'

with open(FIXTURES_PATH / 'single_thread.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = run_train_bpe.run_train_bpe(input_path=input_path, vocab_size=10000, special_tokens=["<|endoftext|>"])
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)
    
with open(FIXTURES_PATH / 'multi_thread.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    profiler = cProfile.Profile()
    profiler.enable()
    vocab, merges = run_train_bpe.run_train_bpe(input_path=input_path, vocab_size=10000, special_tokens=["<|endoftext|>"], num_processes = 8)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)
    
with open(FIXTURES_PATH / 'vocab.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    for i in vocab:
        print(i, vocab[i])
    
with open(FIXTURES_PATH / 'merges.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    for i in merges:
        print(i[0], i[1])