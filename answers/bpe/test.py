import run_train_bpe

FIXTURES_PATH = '/home/cbacoding/llm/CS336/assignment1-basics/answers/bpe'
input_path = FIXTURES_PATH + '/' + 'test.txt'

# test pretokenize
words = run_train_bpe.pretokenize(input_path, ["<|endoftext|>"])
print(len(words))

# test translator
print(run_train_bpe.translator("Hello"))

# test counter
words_list = run_train_bpe.word_initializer(words)
cnt = run_train_bpe.counter(list(words.keys()), words, words_list)
print(len(cnt))

# test selector
print(run_train_bpe.selector(cnt))
check = {
    (123, 321): 1, (321, 123): 1, (111, 333): 1, (333, 111): 1, 
    (123, 21): 2, (21, 123): 2, (11, 333): 2, (333, 11): 2
}
print(run_train_bpe.selector(check))

vocab, merges = run_train_bpe.run_train_bpe(
    input_path=input_path,
    vocab_size=1000,
    special_tokens=["<|endoftext|>"],
)