import json
import os
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import BinaryIO
import regex as re
import click

# for GPT-2
PRE_TOKENIZE_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(
    fp: Path,
    start: int,
    end: int,
    special_tokens: list[str],
    q_done: multiprocessing.Queue,
):
    """
    This function is supposed to run in a separated worker process.

    Spec:
        1. Read the binary chunk data specified by start and end index from file
        2. Decode the chunk into unicode string
        3. Split the resulting text by given special tokens, then for each
            piece from the split (not including any special tokens),
            pre-tokenize it and save resulting pre-tokens and corresponding
            counts to a Counter.
        4. After processing of all pieces done, send Counter value to queue.

    Open the file in binary mode to avoid unnecessary decoding overhead. Only
    decode when necessary.
    """
    with open(fp, "rb") as f:
        f.seek(start)
        # chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = f.read(end - start).decode("utf-8", errors="strict")
    # Stories are separated by special tokens,
    # remove special tokens before pretokenization
    special_tokens_pattern = re.compile("|".join(re.escape(t) for t in special_tokens))
    # NOTE per inspection in data I see there can exist lots of new lines
    # before each text piece begins; IMO we can trim such spaces away
    pieces_to_pretokenize = [
        p.lstrip() for p in re.split(special_tokens_pattern, chunk)
    ]
    pre_token_cnts = Counter()
    for p in pieces_to_pretokenize:
        # makes no sense to consider overlapped matches as it will produce
        # tons of duplicated pre-tokens.
        for m in re.finditer(PRE_TOKENIZE_PAT, p):
            pre_token_cnts[p[m.start() : m.end()]] += 1

    q_done.put(pre_token_cnts)


@click.command(name="pretokenizer")
@click.argument("infile", type=click.Path(exists=True, path_type=Path))
@click.argument("outfile", type=click.Path(exists=False, path_type=Path))
def run(infile: Path, outfile: Path):
    """
    Spec:

        For each chunk:
            Start a worker process doing following:
                1. Pre-tokenization that chunk
                2. Return pre-tokenization output (type?) to a queue
            Receive per-chunk output from queue and aggregate to a final output
            Save the final output to a file on disk.

    Performance:

    $ time python3 cs336_basics/pretokenizer.py data/TinyStoriesV2-GPT4-train.txt  data/TinyStoriesV2-GPT4-train.txt.pretokens.json

    1093.03s user 62.86s system 615% cpu 3:07.92 total
    """
    with open(infile, "rb") as f:
        num_processes = os.process_cpu_count()
        assert num_processes is not None
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Parallelize this by sending each start/end pair to a set of processes.
    num_chunks = len(boundaries) - 1
    q_done = multiprocessing.Queue(num_chunks)
    for start, end in zip(boundaries, boundaries[1:]):
        multiprocessing.Process(
            target=pretokenize, args=(infile, start, end, ["<|endoftext|>"], q_done)
        ).start()

    # aggregate pre token frequency count from each chunk
    final_pre_token_cnts = Counter()
    for _ in range(num_chunks):
        final_pre_token_cnts += q_done.get()

    with open(outfile, "wt", encoding="utf8") as f:
        json.dump(final_pre_token_cnts, f)


if __name__ == "__main__":
    run()
