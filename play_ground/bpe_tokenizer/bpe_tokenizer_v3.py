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
    
    def merge_two_node(self, first_node, second_node):
        assert first_node.next is second_node, "Merged nodes must be adjacent"
        assert first_node is not self.dll.head and second_node is not self.dll.tail, "Cannot merge with sentinel nodes"
        merge_val = first_node.val + second_node.val
        node = _Node(merge_val)
        node.prev, node.next = first_node.prev, second_node.next
        first_node.prev.next = node 
        second_node.next.prev = node
        if merge_val not in self.map:
            self.map[merge_val] = []
        self.map[merge_val].append(node)
        
        # delete node
        val1_nodes = self.map[first_node.val]
        val1_nodes.remove(first_node)
        if not val1_nodes:  # 如果列表空了
            del self.map[first_node.val] # 就从 map 中移除这个键
        val2_nodes = self.map[second_node.val]
        val2_nodes.remove(second_node)
        if not val2_nodes:
            del self.map[second_node.val]
        self.length -= 1

    
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
            special_split_tokens: list[bytes]
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
            
            found_at = -1
            for token in special_split_tokens:
                temp_found = mini_chunk.find(token)
                if temp_found != -1 and found_at != -1 and temp_found < found_at:
                    found_at = temp_found
            if found_at != -1:
                chunk_boundaries[i] = inital_position + found_at
                break

            inital_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def pre_tokenization(chunk: bytes, special_tokens) -> list[bytes]:

    special_split_token_pat =  '|'.join([re.escape(token) for token in special_tokens])
    pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|<\|endoftext\|>"""
    
    text_chunk = chunk.decode("utf-8")
    splited_chunks = re.split(special_split_token_pat, text_chunk)
    words = []
    for splited_chunk in splited_chunks:
        words.extend(re.findall(pat, splited_chunk))
    return words

def convert_and_count(words: list[str]) -> tuple[IndexLinkedList, Counter]:
    ill = IndexLinkedList()
    count = Counter()
    for word in words:
        print(word)
        for byte, byte_second in zip(word, word[1:]):
            print(byte, byte_second)
            byte = byte.encode()
            byte_second = byte_second.encode()
            ill.put(byte)
            merge_val = (byte, byte_second)
            # print(merge_val)
            count.update([merge_val])
        # print('\n')
        ill.put(word[-1].encode())
        ill.put(None)
    return ill, count

# 优化 1.3: 使用 collections.Counter 优化计数
def count_from_chunks(ill: IndexLinkedList) -> Counter:
    counts = Counter()
    first_node, second_node = ill.get_head_node().next, ill.get_head_node().next.next
    while first_node.val is not None or second_node.val is not None:
        if first_node.val is not None and second_node.val is not None:
            merge_val = (first_node.val, second_node.val)
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

def update_ill(ill: IndexLinkedList, count: Counter, max_pair: tuple[Token, Token]) -> tuple[IndexLinkedList, Counter]:
    first_val = max_pair[0]
    second_val =  max_pair[1]
    merged_val = first_val + second_val
    del count[(first_val, second_val)]
    first_list = ill.map[first_val]
    for first_node in first_list:
        if first_node.next is not None and first_node.next.val == second_val:
            second_node = first_node.next
            prev_node, next_node = first_node.prev, second_node.next
            if prev_node.val:
                del_prev = (prev_node.val, first_node.val)
                add_prev = (prev_node.val, merged_val)
                count.subtract([del_prev])
                count.update([add_prev])
            if next_node.val:
                del_next = (second_node.val, next_node.val)
                add_next = (merged_val, next_node.val)
                count.subtract([del_next])
                count.update([add_next])
            ill.merge_two_node(first_node, second_node)
    return ill, count
    

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
    
    desired_num_chunks = multiprocessing.cpu_count()  # 使用CPU核心数作为默认分块数
    # desired_num_chunks = 10  # 使用CPU核心数作为默认分块数
    special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]

    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, special_tokens_bytes)
        chunks = []
        for i in range(len(chunk_boundaries) - 1):
            start, end = chunk_boundaries[i], chunk_boundaries[i+1]
            f.seek(start)
            chunks.append(f.read(end - start))

    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token_bytes in enumerate(special_tokens_bytes):
        vocab[256+i] = special_token_bytes
    merges = []
    with multiprocessing.Pool(desired_num_chunks) as pool:
        args_for_starmap = zip(chunks, repeat(special_tokens))
        results = pool.starmap(pre_tokenization, args_for_starmap)
    words = []
    for result in results:
        print(result)
        words.extend(result)
    print(type(words))
    print(type(words[0]))
    ill, count = convert_and_count(words)
    print(f"count: {count}")
    num_merges = vocab_size - len(vocab)
    for i in tqdm(range(num_merges), desc="Training BPE"):

        max_pair = max(count, key=lambda k: (count.get(k), k))
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        ill, count = update_ill(ill, count, max_pair)
        print(max_pair)
        print(ill.get_length())
    print(merges)
    return vocab, merges
    

if __name__ == '__main__':
    # 确保在 if __name__ == '__main__': 下运行，这是 multiprocessing 的最佳实践
    run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 300, ['<|endoftext|>'])