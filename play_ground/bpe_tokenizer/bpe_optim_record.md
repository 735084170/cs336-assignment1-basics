# BPE tokenizer completion

## version 1

naive completion of bpe tokenizer

- file name: play_ground/bpe_tokenizer
- dataset: data/TinyStoriesV2-GPT4-valid.txt'
- vocab_size: 300
- cost time: 7:28

## version 2

optim by gemini 2.5 pro 

- file name: play_ground/bpe_tokenizer_v2
- dataset: data/TinyStoriesV2-GPT4-valid.txt'
- vocab_size: 300
- cost time: 4:23
- memory: 145M


## version 3

使用hash记录每个pair的索引，只更新最近merge相关的内容

- file name: play_ground/bpe_tokenizer_v3
- dataset: data/TinyStoriesV2-GPT4-valid.txt'
- vocab_size: 300
- cost time:
- memory： 