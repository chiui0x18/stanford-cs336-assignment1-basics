import pickle
import json
from pathlib import Path
import click
from cs336_basics.pretokenizer import pretokenize
from cs336_basics.bpe import UTF8, train_bpe


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
    Pretokenize given text corpus and save output to disk.

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

    with open(outfile, "wt", encoding="utf8") as f:
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
    Train Binary Pair Encoding tokenizer and persist the resultant vocabulary and merged token pairs.

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


if __name__ == "__main__":
    app()
