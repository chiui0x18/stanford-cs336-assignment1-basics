import pickle
import json
from pathlib import Path
from typing import IO
import numpy as np
import numpy.typing as npt
import click
from cs336_basics.pretokenizer import pretokenize
from cs336_basics.bpe import UTF8, train_bpe, Tokenizer
from cs336_basics.log import get_logger


log = get_logger("cli")


# For hierarchy of this CLI app see
# https://click.palletsprojects.com/en/stable/commands-and-groups/
@click.group("lmfun")
def app():
    """
    CLI of language modeling mechanisms built in this assignment.
    """
    pass


@app.command(name="nop")
def nop():
    """
    No-op command entry point to experiment w/ click
    """
    print("nop!")


@app.command(name="pretokenize")
@click.argument("infile", type=click.Path(exists=True, path_type=Path))
@click.argument("outfile", type=click.Path(exists=False, path_type=Path))
@click.argument("CT", type=str)
@click.option(
    "--special-tokens",
    "-s",
    multiple=True,
    help="Special tokens to remove before pretokenizing the text."
    ' e.g. -s "<|endoftext|>" -s "<|im_sep|>"',
)
def cli_pretokenize(
    infile: Path, outfile: Path, ct: str, special_tokens: tuple[str, ...]
):
    """
    Pretokenize text corpus.

    INFILE File path of text corpus to pretokenize.

    OUTFILE Path to the file that will contain pretokenization output.

    CT Special token to chunk input text corpus to speed up pretokenization.
    """
    # Spec:
    #     For each chunk:
    #         Start a worker process doing following:
    #             1. Pre-tokenization that chunk
    #             2. Return pre-tokenization output (type?) to a queue
    #         Receive per-chunk output from queue and aggregate to a final output
    #         Save the final output to a file on disk.
    #
    # Performance:
    # 1093.03s user 62.86s system 615% cpu 3:07.92 total
    # pretoken_cnts = pretokenize(
    #    infile, split_special_token=b"<|endoftext|>", special_tokens=["<|endoftext|>"]
    # )
    pretoken_cnts = pretokenize(
        infile,
        split_special_token=ct.encode(UTF8),
        special_tokens=list(special_tokens),
    )

    with open(outfile, "w", encoding=UTF8) as f:
        json.dump(pretoken_cnts, f)


@app.command(name="bpe")
@click.argument("infile", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "dst", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.argument("size", type=click.INT)
@click.option(
    "--special-tokens",
    "-s",
    multiple=True,
    help="Special tokens to add to the trained token vocabulary."
    ' e.g. -s "<|endoftext|>" -s "<|someothersep|>"',
)
def cli_train_bpe(infile: Path, dst: Path, size: int, special_tokens: tuple[str, ...]):
    """
    Train Binary Pair Encoding tokenizer.

    It saves resultant vocabulary and merged token pairs to `dst`.

    INFILE the file path of mapping of pretokens to their counts in text corpus.

    DST the path to the directory to save trained vocabulary and merged
        pairs. Save both in Python pickle format to file `vocab.pkl` and
        `merges.pkl` respectively.

    SIZE the size of vocabulary that byte-pair encoding logic will build.
    """
    vocab, merges = train_bpe(
        infile,
        vocab_size=size,
        # special_tokens=["<|endoftext|>"],
        special_tokens=list(special_tokens),
    )
    # For simplicity use pickle to serialize both
    with open(dst / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(dst / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)


@app.command(name="tokenize")
@click.argument(
    "cfg", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.argument("txt", type=click.File(mode="r", encoding=UTF8))
@click.argument("out", type=click.Path(exists=False, path_type=Path))
@click.option(
    "--special-tokens",
    "-s",
    multiple=True,
    help="Special tokens to split input text before tokenization. Use the same"
    " list of special tokens for BPE training that produces the input vocabulary"
    ' and merges. e.g. -s "<|endoftext|>" -s "<|someothersep|>"',
)
@click.option(
    "--buffered-tokens",
    "-b",
    type=click.INT,
    default=2**26,
    help="# tokens to buffer in memory before saving to disk."
    " In-memory buffer size in bytes = # tokens buffered x 4",
)
def cli_tokenize(
    cfg: Path,
    txt: IO[str],
    out: Path,
    special_tokens: tuple[str, ...],
    buffered_tokens: int,
):
    """
    Tokenize text per given vocabulary and merges.

    CFG Directory containing tokenizer's vocabulary and merges files (`vocab.pkl` and `merges.pkl`)

    TXT Path to file containing text to tokenize. Specify `-` for stdin.

    OUT Path to save tokenization output. Output will be in form of serialized
    numpy array.

    Each token generated from given text is saved as a 32-bit unsigned int.
    This means it can handle tokenization w/ vocabulary size < 2^32 = 4.29B

    To access tokenized output, use following to minimize mem pressure:

        tokens = np.memmap(f, dtype=np.uint32, mode='r')
    """
    # Spec:
    # Load vocab and merges from path `{cfg}/vocab.pkl` and
    #     `{cfg}/merges.pkl` respectively. Then create tokenizer from the
    #     loaded data.
    # Read text from IN into a generator.
    # Call tokenizer's encode_iterable() to get the token stream.
    #
    # The hardest part of this is to save all the tokens from stream above to
    # a numpy array on disk.
    #
    # To store all the tokens resulting from input text, there must be enough
    # disk space. Because the input text's size can be too large to fit into
    # physical RAM, so reading too many tokens and keeping them in RAM will very
    # likely lead to OOM. Our solution thus needs to balance between 3 aspects:
    # - use of compute resource (cpu and RAM)
    # - use of disk space
    # - use of time to save all the resultant tokens
    #
    # To balance (or limit) the use of compute resuorce (esp. RAM), explicitly
    # control # tokens read into memory; To balance the use of disk space,
    # use numpy's ability to 1/ dynamically expanding the size of array supported
    # by memmap-backed file and 2/ handy support on array concatenation. This
    # also helps narrow down the program running time to only 1 pass over all
    # generated tokens.
    tokenizer = Tokenizer.from_files(
        vocab_fp=cfg / "vocab.pkl",
        merges_fp=cfg / "merges.pkl",
        special_tokens=list(special_tokens),
    )
    token_gen = tokenizer.encode_iterable(txt)
    dtype = np.uint32
    # in-memory buffer
    buf = np.empty(shape=(buffered_tokens,), dtype=dtype)
    final: npt.ArrayLike | None = None
    tokens_read, tokens_buf = 0, 0
    for t in token_gen:
        buf[tokens_buf] = t
        tokens_buf += 1
        tokens_read += 1

        if tokens_buf == buffered_tokens:
            final = _flush_to_disk(final, out, buf, tokens_buf, dtype)
            # all tokens in buf saved. Reuse buf by rewinding count of tokens
            # read in buf
            tokens_buf = 0

    if tokens_read == 0:
        log.info("No token generated. Nothing to do")
        return
    # Chances are there still exist tokens read to buf but not yet flushed to
    # disk. So flush once more.
    final = _flush_to_disk(final, out, buf, tokens_buf, dtype)
    log.info(f"Encoded input text into {tokens_read} tokens")


def _flush_to_disk(
    arr: npt.NDArray | None,
    fp: Path,
    buf: npt.NDArray,
    tokens_buf: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray:
    """
    Flush data in given buffer array to file on disk pointed by arr and fp.

    arr: np memmap or None. Associated w/ the file on disk where data in buf
        will be flushed to.
    fp: Path to the file on disk where data in buf will be flushed to.
    buf: np array holding buffered data.
    return: the updated np memmap reference.
    """
    first_available_idx = arr.shape[0] if arr is not None else 0
    new_shape = first_available_idx + tokens_buf
    try:
        if arr is None:
            arr_ = np.memmap(fp, dtype=dtype, mode="w+", shape=(new_shape,))
        else:
            # Append data to end of file. Must allocate enough space first.
            arr_ = np.memmap(fp, dtype=dtype, mode="r+", shape=(new_shape,))
    except:
        # clean up memmap file on disk in case of err
        fp.unlink(missing_ok=True)
        log.exception("Error creating numpy memmap-backed array")
        raise

    arr_[first_available_idx:] = buf[:tokens_buf]
    arr_.flush()
    return arr_


if __name__ == "__main__":
    app()
