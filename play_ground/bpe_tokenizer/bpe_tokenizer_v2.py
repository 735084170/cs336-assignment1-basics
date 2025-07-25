"""
    naive 实现，经过genimi优化
"""


import os
from typing import IO, BinaryIO, TypeAlias
from tqdm import tqdm
import regex as re
import multiprocessing
from itertools import repeat
from collections import Counter

# 优化 2.2: 使用 TypeAlias 增加代码可读性
Token = bytes
TokenGroup = list[Token]
TokenGroupList = list[TokenGroup]
TokenGroupBatch = list[TokenGroupList]

def find_special_tokens(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read()
    pattern = r'<\|(.*?)\|>'
    return list(set(re.findall(pattern, text)))

def find_chunk_boundaries(
            file: BinaryIO,
            desired_num_chunks: int,
            special_split_token: bytes
    ) -> list[int]:
        
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    mini_chunk_size = 4096

    chunk_boundaries = []
    for i in range(0, desired_num_chunks+1):
        chunk_boundaries.append(chunk_size*i)
    chunk_boundaries[-1] = file_size

    for i in range(1, desired_num_chunks-1):
        inital_position = chunk_boundaries[i]
        file.seek(inital_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[i] = file_size
                break
            
            found_at = mini_chunk.find(special_split_token)
            if found_at != -1:
                chunk_boundaries[i] = inital_position + found_at
                break

            inital_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pre_tokenization(chunk: bytes) -> list[bytes]:

    special_split_token_pat = r'<\|endoftext\|>'
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|<\|endoftext\|>"""
    
    text_chunk = chunk.decode("utf-8")
    splited_chunks = re.split(special_split_token_pat, text_chunk)
    blocks = []
    for splited_chunk in splited_chunks:
        blocks.extend(re.findall(pat, splited_chunk))
    bytes_blocks = [block.encode("utf-8") for block in blocks]
    return bytes_blocks

def convert_to_vocab(blocks: list[Token]) -> TokenGroupList:
    return [[bytes([byte]) for byte in block] for block in blocks]

# 优化 1.3: 使用 collections.Counter 优化计数
def count_from_chunks(token_group_list: TokenGroupList) -> Counter:
    counts = Counter()
    for token_group in token_group_list:
        # 使用 zip 可以更优雅地创建词对
        counts.update(zip(token_group, token_group[1:]))
    return counts

def update_token_group_list(token_group_list: TokenGroupList, max_pair: tuple[Token, Token]) -> TokenGroupList:
    new_token_group_list = []
    new_token = max_pair[0] + max_pair[1]
    
    for token_group in token_group_list:
        new_group = []
        i = 0
        while i < len(token_group):
            if i < len(token_group) - 1 and (token_group[i], token_group[i+1]) == max_pair:
                new_group.append(new_token)
                i += 2
            else:
                new_group.append(token_group[i])
                i += 1
        new_token_group_list.append(new_group)
    return new_token_group_list

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    desired_num_chunks = multiprocessing.cpu_count()  # 使用CPU核心数作为默认分块数
    special_split_token = b"<|endoftext|>"

    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, special_split_token)
        chunks = []
        for i in range(len(chunk_boundaries) - 1):
            start, end = chunk_boundaries[i], chunk_boundaries[i+1]
            f.seek(start)
            chunks.append(f.read(end - start))

    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = special_split_token
    merges = []

    # 初始化分词
    token_group_list_batch = [convert_to_vocab(pre_tokenization(chunk)) for chunk in chunks]
    
    # 优化 1.1: 在循环外创建并复用进程池
    with multiprocessing.Pool() as pool:
        num_merges = vocab_size - len(vocab)
        for i in tqdm(range(num_merges), desc="Training BPE"):
            # 并行计数
            # 使用 pool.map 高效并行处理
            all_counts = pool.map(count_from_chunks, token_group_list_batch)
            
            # 优化 1.3: 使用 Counter 高效聚合结果
            total_counts = sum(all_counts, Counter())

            if not total_counts:
                print("No more pairs to merge.")
                break

            # 优化 2.3: 使用 max() 和 key 简化查找
            max_pair = max(total_counts, key=total_counts.get)
            
            vocab[len(vocab)] = max_pair[0] + max_pair[1]
            merges.append(max_pair)

            # 并行更新 token 列表
            args_for_starmap = zip(token_group_list_batch, repeat(max_pair))
            # starmap 会自动解包元组，将元素作为独立参数传入
            token_group_list_batch = pool.starmap(update_token_group_list, args_for_starmap)

    print(f"\nFinal vocab size: {len(vocab)}")
    return vocab, merges

if __name__ == '__main__':
    # 确保在 if __name__ == '__main__': 下运行，这是 multiprocessing 的最佳实践
    run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 300, [])