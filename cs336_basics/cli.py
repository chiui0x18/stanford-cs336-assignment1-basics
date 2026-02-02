from datetime import datetime
import pickle
import json
from pathlib import Path
from typing import IO
import numpy as np
import numpy.typing as npt
import torch
import click
from cs336_basics.pretokenizer import pretokenize
from cs336_basics.bpe import UTF8, train_bpe, Tokenizer
from cs336_basics.log import get_logger
from cs336_basics.train import train_loop


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
    time_start = time.time()
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
    duration_seconds = time.time() - time_start
    log.info(
        f"Encoded input text into {tokens_read} tokens in" f" {duration_seconds:.2f}s"
    )


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
    # Append tokens buffered to end of saved token sequence and flush the
    # change to disk
    arr_[first_available_idx:] = buf[:tokens_buf]
    arr_.flush()
    return arr_


@app.command(name="train", context_settings={"show_default": True})
@click.argument(
    "training_data", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "validation_data", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--bsize",
    type=click.INT,
    required=True,
    default=16,
    help="Batch size of training and validation data to sample and use per epoch",
)
@click.option(
    "--vocab-size",
    type=click.INT,
    required=True,
    default=None,
    help="Token vocabulary size. This MUST match the size of token vocabulary"
    " used to encode tokens from raw training and validation text data",
)
@click.option(
    "--ctx-len",
    type=click.INT,
    required=True,
    default=256,
    help="Max length of sequence which the trained Transformer model can"
    " process. TODO: Test what will happen if we feed sequence longer than this"
    " to model",
)
@click.option(
    "--layers",
    type=click.INT,
    required=True,
    default=4,
    help="Number of Transformer blocks to include in model to be trained",
)
@click.option(
    "--d-model",
    type=click.INT,
    required=True,
    default=512,
    help="Size of model's dimension, aka Transformer model's hidden dimension",
)
@click.option(
    "--heads",
    type=click.INT,
    required=True,
    default=16,
    help="Number of heads to use in model's multi-head attention layer",
)
@click.option(
    "--d-ff",
    type=click.INT,
    required=True,
    default=1344,
    help="Size of Transformer model elementwise feed-forward network dimension."
    " Recommend a value around 8/3 of model dimension and a multiple of 64",
)
@click.option(
    "--theta", type=click.FLOAT, default=10000, help="Value of theta for RoPE"
)
@click.option(
    "--tmax",
    type=click.INT,
    required=True,
    default=16,
    help="Max number of epochs the training loop will iterate",
)
@click.option(
    "--twarmup",
    type=click.INT,
    default=16,
    help="Number of epochs to warmup learning rate per cos annealling schedule",
)
@click.option(
    "--tcool",
    type=click.INT,
    default=16,
    help="Number of epochs after which cos annealling learning rate schedule ends",
)
@click.option(
    "--lr-max",
    type=click.FLOAT,
    default=1e-3,
    help="Max learning rate per cos annealling schedule",
)
@click.option(
    "--lr-min",
    type=click.FLOAT,
    default=1e-4,
    help="Min learning rate per cos annealling schedule",
)
@click.option(
    "--grad-clip-norm",
    type=click.FLOAT,
    default=None,
    help="L2 norm threshold to apply gradient clipping",
)
@click.option(
    "--from-checkpoint",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a checkpointed model which training run will resume on",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory to save model checkpoints. Use hierarchical path pattern eg"
    " `./checkpoints/run-my-comments/`. Model checkpoint files will have name"
    " pattern `{run_epoch_num}.pt`",
)
@click.option(
    "--checkpoint-interval",
    type=click.INT,
    default=None,
    help="Interval, in # epochs, to checkpoint model training progress",
)
@click.option(
    "--metric-interval",
    type=click.INT,
    default=20,
    help="Interval, in # epochs, to metric training loss. To facilitate "
    "comparison b/w training and eval loss, it shall divide eval interval value",
)
@click.option(
    "--eval-interval",
    type=click.INT,
    default=50,
    help="Interval, in # epochs, to evaluate trained model w/ validation dataset",
)
@click.option(
    "--rand-seed",
    type=click.INT,
    default=None,
    help="RNG seed for reproducible randomness behavior in debugging. Range in [0, 2^32-1]",
)
@click.option(
    "--device",
    type=click.STRING,
    required=True,
    default="cpu",
    help="device to allocate tensors for model training",
)
@click.option(
    "--dtype",
    type=click.STRING,
    required=True,
    default="float32",
    help="data type used for tensor representation and operations",
)
@click.option(
    "--log-dir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to directory to save Tensorboard event files for monitoring and"
    " dashboarding. Vary this by training run unless resuming from checkpoint"
    " in a particular run. To facilitate comparison between runs, use"
    " hierarchical path structure eg ./runs/paramName1Val1_etc_etc",
)
@click.option(
    "--autograd-detect-anomaly",
    is_flag=True,
    help="Enable Pytorch autograd anomaly detection. For debug only",
)
@click.option(
    "--metric-grad-norm",
    is_flag=True,
    help="Compute and metric model param gradient L2 norm. For debug only",
)
def cli_train(
    training_data,
    validation_data,
    bsize: int,
    vocab_size: int,
    ctx_len: int,
    layers: int,
    d_model: int,
    heads: int,
    d_ff: int,
    theta: float,
    tmax: int,
    twarmup: int,
    tcool: int,
    lr_max: float,
    lr_min: float,
    grad_clip_norm: float,
    from_checkpoint: Path,
    checkpoint_dir: Path,
    checkpoint_interval: int,
    metric_interval: int,
    eval_interval: int,
    rand_seed: int,
    device: str,
    dtype: str,
    log_dir: Path,
    autograd_detect_anomaly: bool,
    metric_grad_norm: bool,
):
    """
    Run training loop for Transformer model of given spec.

    TODO Argument for file path to save training checkpoints including the
    final trained model data.


    autograd_detect_anomaly: bool = False,
    tensorboard_log_dir: str | None = None,
    eval_interval: int = 50,
    """
    cli_args = locals()
    log_dir.mkdir(parents=True, exist_ok=True)
    # Save CLI argument values for bookkeeping purpose
    with open(log_dir / "cli.args.json", "w") as f:
        json.dump(cli_args, f, default=str, indent=2)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype: torch.dtype = torch.float32
    # TODO support other data types
    match dtype:
        case "bfloat16":
            torch_dtype = torch.bfloat16
        case "float16":
            torch_dtype = torch.float16
        case _:
            pass

    train_set = np.memmap(training_data, dtype=np.uint32, mode="r")
    valid_set = np.memmap(validation_data, dtype=np.uint32, mode="r")
    start = datetime.now()
    train_loop(
        train_set=train_set,
        validation_set=valid_set,
        batch_size=bsize,
        vocab_size=vocab_size,
        context_len=ctx_len,
        num_layers=layers,
        d_model=d_model,
        num_heads=heads,
        d_ff=d_ff,
        rope_theta=theta,
        t_max=tmax,
        t_warmup=twarmup,
        t_cool=tcool,
        lr_max=lr_max,
        lr_min=lr_min,
        grad_clip_max_norm=grad_clip_norm,
        from_checkpoint=from_checkpoint,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        rand_seed=rand_seed,
        device=device,
        dtype=torch_dtype,
        tensorboard_log_dir=log_dir,
        eval_interval=eval_interval,
        autograd_detect_anomaly=autograd_detect_anomaly,
        metric_interval=metric_interval,
        metric_grad_norm=metric_grad_norm,
    )
    d_train = datetime.now() - start
    log.info(f"Training run took {d_train}")


if __name__ == "__main__":
    app()
