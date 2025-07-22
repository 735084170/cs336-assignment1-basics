import os
from typing import IO, Any, BinaryIO
from tqdm import tqdm
# import re
import regex as re

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
    # parallel chunks
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
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text_chunk = chunk.decode("utf-8")
        blocks = re.findall(pat, text_chunk)
        bytes_blocks = [block.encode("utf-8") for block in blocks]
        return bytes_blocks

    def convert_to_vocab(blocks: list[bytes]):
        vocab_blocks = []
        for block in blocks:
            vocab_block = [bytes([byte]) for byte in block]
            vocab_blocks.append(vocab_block)
        return vocab_blocks

   
    def count_from_chunks(token_group_list: list):
        merge_count = {}

        for token_group in token_group_list:
            for i in range(len(token_group)-1):
                pre = token_group[i]
                latter = token_group[i+1]
                if (pre, latter) not in merge_count:
                    merge_count[(pre, latter)] = 0
                merge_count[(pre, latter)] += 1
        return merge_count
    
    def update_token_group_list(token_group_list):
        # vocab增加新的合并项，重新处理token_group_list
        update_token_group_list =[]
        for token_group in token_group_list:
            update_token_group = []
            if len(token_group) == 1:
                update_token_group = token_group
            else:
                idx = 0
                while idx < len(token_group):
                    if idx == len(token_group)-1:
                        update_token_group.append(token_group[idx])
                        break
                    pre = token_group[idx]
                    latter = token_group[idx+1]
                    if pre == max_vab[0] and latter == max_vab[1]:
                        update_token_group.append(max_vab[0]+max_vab[1])
                        idx += 2
                    else:
                        update_token_group.append(pre)
                        idx += 1
            update_token_group_list.append(update_token_group)
        
        return update_token_group_list

    desired_num_chunks = 8
    special_split_token = "<|endoftext|>".encode("utf-8")

    # split the corpus to chunk for parallel
    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, special_split_token)
        chunks = []
        for i in range(len(chunk_boundaries)-1):
            start_pos = chunk_boundaries[i]
            end_pos = chunk_boundaries[i+1]
            chunk_size = end_pos - start_pos
            f.seek(start_pos)
            chunk = f.read(chunk_size)
            chunks.append(chunk)

    # initial the vocab 
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = special_split_token

    for chunk in chunks:
        # pre tokenization, split the chunk into words
        words = pre_tokenization(chunk)
        # split the words into token
        token_group_list = convert_to_vocab(words)
        for _ in tqdm(range(vocab_size-len(vocab))):
            # count merge
            merge_count = count_from_chunks(token_group_list)

            # get the new merge vocab
            max_count = 0
            max_vab = (bytes([0]),bytes([0]))
            for k, v in merge_count.items():
                if v > max_count or (v == max_count and k > max_vab):
                    max_count = v
                    max_vab = k
            vocab[len(vocab)] = max_vab[0] + max_vab[1]
            
            # update base on the new vocab
            token_group_list = update_token_group_list(token_group_list)



        break
    print(vocab)

run_train_bpe(
    'data/TinyStoriesV2-GPT4-valid.txt',
    500,
    [])