import answers.bpe.run_train_bpe as run_train_bpe
import pathlib
import sys

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent)
# FIXTURES_PATH = '/home/cbacoding/llm/CS336/assignment1-basics/answers/bpe'
input_path = FIXTURES_PATH / 'test.txt'

with open(FIXTURES_PATH / 'output.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    
    # test pretokenize
    words = run_train_bpe.pretokenize(input_path, ["<|endoftext|>"])
    print("The number of words = ", len(words))

    # test translator
    print(run_train_bpe.translator("Hello"))

    # test counter
    words_list = run_train_bpe.words_map_initializer(words)
    cnt = run_train_bpe.counter(list(words.keys()), words, words_list)
    print("The number of pairs = ",len(cnt))

    # test selector
    print(run_train_bpe.selector(cnt))
    check = {
        (123, 321): 1, (321, 123): 1, (111, 333): 1, (333, 111): 1, 
        (123, 21): 2, (21, 123): 2, (11, 333): 2, (333, 11): 2
    }
    print(run_train_bpe.selector(check))

    import cProfile
    import pstats

    # 创建一个Profiler对象
    profiler = cProfile.Profile()

    # 开始性能分析
    profiler.enable()

    # 运行代码
    vocab, merges = run_train_bpe.run_train_bpe(input_path=input_path, vocab_size=400, special_tokens=["<|endoftext|>"])
    # cnt = run_train_bpe.counter(list(words.keys()), words, words_list)

    # 结束性能分析
    profiler.disable()

    # 使用pstats模块将二进制文件转换成文本文件
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(10)