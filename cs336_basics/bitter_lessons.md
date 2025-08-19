# bitter lessons

## run_train_bpe

- 暴力很可能非常耗时
- 多线程不一定速度快
  不要尝试在没有完整写出单线程代码的前提下书写多线程代码

# tokenizer

- 注意合并的顺序，贪心优化可能导致合并顺序出错
- 若某些操作需要重复多次可以缓存对应的结果

# attention

- 使用einsum时要小心，必须将含义相同的字符串修改为相同的字符串
return einsum(attn_probs, V, " ... queries keys,  ... values d_v ->  ... queries d_v")