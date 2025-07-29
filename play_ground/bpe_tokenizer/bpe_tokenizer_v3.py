"""
    增量实现
    记录每个pair的位置，只更新刚刚merge相关内容
"""

import os
from typing import IO, BinaryIO, TypeAlias
from tqdm import tqdm
import regex as re
import multiprocessing
from itertools import repeat
from collections import Counter, deque
import time

# 优化 2.2: 使用 TypeAlias 增加代码可读性
Token = bytes
TokenGroup = list[Token]
TokenGroupList = list[TokenGroup]
TokenGroupBatch = list[TokenGroupList]

class _Node:
    __slots__ = ("val", "prev", "next")
    def __init__(self, val=None):
        self.val = val
        self.prev = self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head, self.tail = _Node(), _Node()
        self.head.next, self.tail.prev = self.tail, self.head

    def append_tail(self, node):
        last = self.tail.prev
        last.next = self.tail.prev = node
        node.prev, node.next = last, self.tail
    
    def remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def pop_head(self):
        if self.head.next is self.tail:
            return None
        node = self.head.next
        self.remove(node)
        return node

class IndexLinkedList:
    def __init__(self):
        self.map = {}
        self.dll = DoublyLinkedList()
        self.length = 0
    
    def put(self, val):
        # val 是 None，表示断开
        node = _Node(val)
        self.dll.append_tail(node)
        if val not in self.map:
            self.map[val] = []
        self.map[val].append(node)
        self.length += 1

    def get_nodes_by_value(self, val):
        return self.map[val]
    
    def insert_between(self, first_node, second_node, new_val):
        node = _Node(new_val)
        if new_val not in self.map:
            self.map[new_val] = []
        self.map[new_val].append(node)
        while first_node.next != second_node:
            cur_node = first_node.next
            self.dll.remove(cur_node)
            self.map.get(cur_node.val, []).remove(cur_node)
            self.length -= 1
        first_node.next = second_node.prev = node

    
    def get_head_node(self):
        return self.dll.head
    
    def get_length(self):
        return self.length



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
    ill = IndexLinkedList()
    for block in blocks:
        for byte in block:
            ill.put(bytes([byte]))
        ill.put(None)
    return ill

# 优化 1.3: 使用 collections.Counter 优化计数
def count_from_chunks(ill: IndexLinkedList) -> Counter:
    counts = Counter()
    first_node, second_node = ill.get_head_node().next, ill.get_head_node().next.next
    while first_node.val is not None or second_node.val is not None:
        if first_node.val is not None and second_node.val is not None:
            merge_val = first_node.val + second_node.val
            counts.update([merge_val])
        first_node = second_node
        second_node = first_node.next
    return counts

def get_count_and_hash(token_group_list: TokenGroupList) -> tuple[Counter, dict[tuple[Token, Token], list]]:
    counts = Counter()
    hash_dict = {}
    for group_idx, token_group in enumerate(token_group_list):
        inner_index = 0
        for pre, latter in zip(token_group[:-1], token_group[1:]):
            counts[(pre, latter)] += 1
            if (pre, latter) not in hash_dict:
                hash_dict[(pre, latter)] = []
            hash_dict[(pre, latter)].append(group_idx, inner_index)
            inner_index += 1
    return counts, hash_dict

def update_ill(ill: IndexLinkedList, count: Counter, max_pair: tuple[Token, Token]) -> TokenGroupList:
    first_val = bytes([max_pair[0]])
    second_val =  bytes([max_pair[1]])
    merged_val = first_val + second_val
    print(f"merged_val:{merged_val}")
    first_list = ill.map[first_val]
    for first_node in first_list:
        if first_node.next.val == second_val:
            second_node = first_node.next
            prev_node, next_node = first_node.prev, second_node.next
            # print(prev_node.val, first_node.val, second_node.val, next_node.val)
            if prev_node.val:
                del_prev = prev_node.val + first_node.val
                add_prev = prev_node.val + merged_val
                count.subtract([del_prev])
                count.update([add_prev])
            if next_node.val:
                del_next = second_node.val + next_node.val
                add_next = merged_val + next_node.val
                count.subtract([del_next])
                count.update([add_next])
            ill.insert_between(prev_node, next_node, merged_val)
 
    return [ill, count]
    

def update_pair_count_and_token_group_list_batch(
        count: Counter, 
        hash_dict: dict[tuple[Token, Token], list],
        token_group_list: TokenGroupList,
        max_pair: tuple[Token, Token]) -> TokenGroupList:
    hash_dict
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
    print(f"desired_num_chunks", desired_num_chunks)
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

    for chunk in chunks:
        ill = convert_to_vocab(pre_tokenization(chunk))
        # 使用 pool.map 高效并行处理
        count = count_from_chunks(ill)

        num_merges = vocab_size - len(vocab)
        for i in tqdm(range(num_merges), desc="Training BPE"):
            print(ill.get_length())


            # 优化 2.3: 使用 max() 和 key 简化查找
            max_pair = max(count, key=count.get)
            
            vocab[len(vocab)] = max_pair[0] + max_pair[1]
            merges.append(max_pair)

            ill, count = update_ill(ill, count, max_pair)


        print(f"\nFinal vocab size: {len(vocab)}")
        return vocab, merges

if __name__ == '__main__':
    # 确保在 if __name__ == '__main__': 下运行，这是 multiprocessing 的最佳实践
    run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 300, [])