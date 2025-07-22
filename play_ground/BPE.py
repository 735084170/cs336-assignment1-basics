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
    

    def pre_tokenization(chunk: bytes):
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text_chunk = chunk.decode("utf-8")
        blocks = re.findall(pat, text_chunk)
        bytes_blocks = [block.encode("utf-8") for block in blocks]
        return bytes_blocks


    desired_num_chunks = 1000
    special_split_token = "<|endoftext|>".encode("utf-8")
    with open(input_path, 'rb') as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks, special_split_token)
        # print(chunk_boundaries)
        
        chunks = []
        for i in range(len(chunk_boundaries)-1):
            start_pos = chunk_boundaries[i]
            end_pos = chunk_boundaries[i+1]
            chunk_size = end_pos - start_pos
            f.seek(start_pos)
            chunk = f.read(chunk_size)
            chunks.append(chunk)
    vocab = {i: bytes([i]) for i in range(256)}
    merge_count = {}
    for chunk in chunks:
        blocks = pre_tokenization(chunk)
        print(chunk[:50])
        while len(vocab) < vocab_size:
            for block in blocks:
                print(block)


            for pre, latter in tqdm(zip(chunk[:-1], chunk[1:])):
                pre = bytes([pre])
                latter = bytes([latter])
                if (pre, latter) not in merge_count:
                    merge_count[(pre, latter)] = 0
                merge_count[(pre, latter)] += 1


            max_count = 0
            max_vab = (bytes([0]),bytes([0]))
            for k, v in merge_count.items():
                if v > max_count or (k == max_vab and k > max_vab):
                    max_count = v
                    max_vab = k
            vocab[len(vocab)] = max_vab[0] + max_vab[1]
                                 




        break
    print(vocab)

run_train_bpe(
    '/home/ubuntu/liyang/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt',
    300,
    [])