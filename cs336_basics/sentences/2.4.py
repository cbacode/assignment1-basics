import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
chunk = "some text that i'll pre-tokenize"
print(re.findall(PAT, chunk))

keys = re.finditer(PAT, chunk)
print(keys)
for match in keys:
    print(match.group())