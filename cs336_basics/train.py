import time
import random
from pathlib import Path
import json
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable, Callable, Iterator
from typing import BinaryIO, IO
import os
import torch
import torch._dynamo
from torch.optim import Optimizer
from torch import Tensor, nn, linalg, autograd
from torch.utils.tensorboard.writer import SummaryWriter
from jaxtyping import Float, Int64

from cs336_basics.log import get_logger
from cs336_basics.transformer import TransformerModel

# NOTE Random number gen to generate batch of randomly sampled token sequences.
_rng = np.random.default_rng()

log = get_logger("Train")


def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int64[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
    """
    input: Logits predicted by model. input[i] are the unormalized scores representing
        the probability distribution of next item for the i-th sequence in given batch.
    targets: Indices of the ground true next item of each sequence in given batch.
    return: Cross entroy loss value across all sequences in given batch.

    Naive impl: Inefficient mem use, does not work well w/ Pytorch middleware and
    hardware level parallelism machinary.
    ```
    maxes, _ = inputs.max(dim=-1, keepdim=True)
    inputs_demaxed = inputs - maxes
    # reduce by sum over exp on last dim
    lg_inputs_exp_sum = torch.log(torch.exp(inputs_demaxed).sum(dim=-1))
    return (
        lg_inputs_exp_sum
        # More see "Indexing with multidimensional index arrays" in
        # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
        - inputs_demaxed[torch.arange(0, inputs_demaxed.shape[0]), targets]
    ).mean()
    ```

    Below is the impl per prompting against ChatGPT, for:
        - Numerical stability w/ Pytorch builtins
        - More use of existing fused kernels for speedier execution on GPUs
        - Lower mem and data move/bandwidth pressure by using less intermediate tensors
        - Simpler autograd graph for speedier backprop
        - Idiomatic Pytorch code
    """
    # shape (batch_size,)
    logsumexp = torch.logsumexp(inputs, dim=-1)
    # torch.gather results in tensor of shape (batch_size 1) so
    # squeeze out the last dim for subsequent subtraction
    # Resultant tensor shape (batch_size,)
    ground_truth_item_logits = torch.gather(
        inputs,
        dim=-1,
        index=targets.unsqueeze(dim=-1),
    ).squeeze(dim=-1)
    # Take avg along the dimension of (sequence length if provided)
    # and batch size, per equation (16) in assignment
    return (logsumexp - ground_truth_item_logits).mean()


def perplexity(
    inputs: Float[Tensor, "seq_len vocab_size"],
    targets: Int64[Tensor, "seq_len"],
) -> Float[Tensor, ""]:
    """
    Compute perplexity metric of a given sequence.
    NOTE This metric is not applicable to a batch of sequences; If given inputs and
    targets has prefixing batch dimension, then call to this function returns a
    1-D tensor, each item of which is the perplexity metric value of each sequence
    in the batch.

    For definition of Perplexity metric see section 3.2.2 of
    [Hosseini et al. (2023)](https://arxiv.org/pdf/2301.09211)

    input: Logits predicted by model. input[i] are the unormalized scores representing
        the probability distribution of next item for the i-th sequence in given batch.
    targets: The true next token of each sequence in given batch.

    return: A scalar tensor if inputs and targets representing a particular
    sequence, or a 1-D tensor if there exists a prefixing batch dimension.
    """
    # shape (seq_len,)
    logsumexp = torch.logsumexp(inputs, dim=-1)
    # torch.gather results in tensor of shape (seq_len 1) so
    # squeeze out the last dim for subsequent subtraction
    # resultant tensor shape (seq_len,)
    ground_truth_item_logits = torch.gather(
        inputs, dim=-1, index=targets.unsqueeze(dim=-1)
    ).squeeze(dim=-1)

    return (logsumexp - ground_truth_item_logits).mean(dim=-1).exp()


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        lr_scheduler: Callable[[Tensor], Tensor] | None = None,
        wd_scheduler: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """
        lr: learning rate
        weight_decay: weight decay rate
        lr_scheduler: learning rate scheduler, aka a function that computes the learning rate to use at training run epoch t.
            IMO closure is the best impl for such schedulers.
        wd_scheduler: weight decay rate scheduler.
        """
        defaults = {
            "lr": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Below presents a naive AdamW implementation.
        It is correct, however neither Pytorch idiomatic nor performant.

        Per ChatGPT, hardware accelerators excel when all math ops are expressed as tensor operations.

        TODO: Refactor for performance and being more idiomatic.
        """
        loss = None if closure is None else closure()
        # What are self.param_groups and self.state? See
        # https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html#torch.optim.Optimizer.state_dict
        for param_group in self.param_groups:
            lr = param_group["lr"]
            beta1, beta2 = param_group["beta1"], param_group["beta2"]
            eps = param_group["eps"]
            weight_decay = param_group["weight_decay"]
            # p of type torch.nn.parameter.Parameter
            for p in param_group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = torch.tensor(1, device=p.device, dtype=torch.float32)
                    state["m"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                t: Tensor = state.get("t")
                m: Tensor = state.get("m")
                v: Tensor = state.get("v")

                # paradigm commented out below is discouraged in Pytorch
                # grad = p.grad.data
                with torch.no_grad():
                    grad = p.grad
                    # use in-place tenor ops to avoid unnecessary mem pressure
                    # due to creation of intermediate tensors
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    # Compute lr w/ scheduler if present
                    if self.lr_scheduler is not None:
                        lr_t = self.lr_scheduler(t)
                    else:
                        lr_t = lr * torch.sqrt(1 - beta2**t) / (1 - beta1**t)

                    if self.wd_scheduler is not None:
                        wd_t = self.wd_scheduler(t)
                    else:
                        wd_t = weight_decay
                    # Apply decoupled weight decay then descent.
                    # Apply weight decay first; Note this aligns w/ the
                    # algo 2 in the original AdamW paper (See https://arxiv.org/pdf/1711.05101) and spec in Pytorch doc
                    # For now we use the same formula with assignment aka scaling weight decay rate by factor of initial lr.
                    # NOTE a different way per the original paper is scaling weight decay rate by factor of current-step lr.
                    p.mul_(1 - lr * wd_t)
                    p.addcdiv_(m, torch.sqrt(v).add_(eps), value=-lr_t)

                t.add_(1)

        return loss


def cosine_annealing_lr_scheduler(
    lr_max: float,
    lr_min: float,
    t_warm: int,
    t_cool: int,
) -> Callable[[Tensor], Tensor]:
    """
    Implements cosine annealing learning rate scheduler mechanism. Its return
    value is a scheduler backed by a callable which takes the current training
    step and returns the learning rate to use at the current training step.

    step: Scalar tensor representing training run step progress.

    Spec:

    It seems way better to use a closure to implement this, so that we
    don't have to pass fixed params eg min and max learning rate over and
    over when it is used.

    To visualize the change of learning rate over # iterations:
    ```
    from cs336_basics.train import cosine_annealing_lr_scheduler
    import matplotlib.pyplot as plt
    import torch

    lr_scheduler = cosine_annealing_lr_scheduler(
                        lr_max=1e-1, lr_min=1e-3, t_warm=100, t_cool=900)

    iters = torch.arange(1, 1001).to(torch.float32)
    lrs = [lr_scheduler(v) for v in iters]

    fig, ax = plt.subplots()
    ax.set_xlabel('# iterations')
    ax.set_ylabel('learning rate')
    ax.plot(iters.numpy(), lrs)
    plt.show()
    ```
    """

    def scheduler(t: Tensor) -> Tensor:
        """
        t: Current training run step represented by a scalar tensor.
        """
        if t < t_warm:
            return (t / t_warm).mul_(lr_max)
        elif t <= t_cool:
            cos_eff: Tensor = (
                (t - t_warm).mul_(torch.pi).div_(t_cool - t_warm).cos_().add_(1)
            )
            return cos_eff.mul_(lr_max - lr_min).mul_(0.5).add_(lr_min)
        else:
            return torch.tensor(lr_min, device=t.device, dtype=t.dtype)

    return scheduler


def cos_anneal_lr_scheduler_flx(
    lr_max: float,
    lr_cool: float,
    lr_cold: float,
    t_warm: int,
    t_cool: int,
    t_cold: int,
) -> Callable[[Tensor], Tensor]:
    """
    Cosine annealing lr scheduler which incorporates my observations in training run.

    The schedule is as follow:
        - phase [t_start, t_warm]: linear increment proportional to epoch
        - phase [t_warm, t_cool]: cosine annealing decrement at scale of [lr_cool, lr_max]
        - phase [t_cool, t_cold]: cosine annealing decrement at scale of [lr_cold, lr_cool]
            - This scale is much smaller than [lr_cool, lr_max]
        - phase [t_cold, t_max]: lr stays constant at lr_cold

    This is to employ a smaller but continuously changing lr in mid- and late-phase of
        training run to facilitate effective descent of loss function as training
        progresses through these phases.

    lr_max: Max lr to use. lr will reach this value after warmup.
    lr_cool: lr at training step t_cool after cosine annealing descent from lr_max.
    lr_cold: lr at training step t_cold after cosine annealing descent from lr_cool.
    t_warm: Training epoch # where warmup ends.
    t_cool: Training epoch # where lr cools down from lr_max to lr_cool.
    t_cold: Training epoch # where lr moves (slowly) from lr_cool to lr_cold.

    For training a tiny Transformer model like that in assignment, confirmed setup eg
        following can lead to effective loss descent over training run:
        - 10k training steps
        - lr warmups to 4e-3 then cosine annealing decreases to 1e-3 at the
            1st 5k training epochs
            - A small fraction of training steps e.g. 20 suffice to warmup
        - lr cosine annealing decreases from 1e-3 to 2.5e-4 at the 2nd 5k
            training epochs
    """
    assert t_warm < t_cool < t_cold

    def scheduler(t: Tensor) -> Tensor:
        if t < t_warm:
            return (t / t_warm).mul_(lr_max)
        elif t <= t_cool:
            cos_eff: Tensor = (
                (t - t_warm).mul_(torch.pi).div_(t_cool - t_warm).cos_().add_(1)
            )
            return cos_eff.mul_(lr_max - lr_cool).mul_(0.5).add_(lr_cool)
        elif t <= t_cold:
            cos_eff: Tensor = (
                (t - t_cool).mul_(torch.pi).div_(t_cold - t_cool).cos_().add_(1)
            )
            return cos_eff.mul_(lr_cool - lr_cold).mul_(0.5).add_(lr_cold)
        else:
            return torch.tensor(lr_cold, device=t.device, dtype=t.dtype)

    return scheduler


def weight_decay_scheduler(
    wd_init: float,
    wd_min: float,
    t_cool: int,
    t_cold: int,
    wd_cool: float | None = None,
) -> Callable[[Tensor], Tensor]:
    """
    wd_init: Initial weight decay rate. Prefer a larger value.
    wd_min: Min weight decay rate to use.
    t_cool: Training step to start decrementing weight decay from wd_init towards wd_min.
    t_cold: Training step after which weight decay rate will stay at wd_min.
    wd_cool: Weight decay rate to use when cooling starts. Typically wd_min < wd_cool <= wd_init

    Spec:

    Apply large weight decay in training run early phase.
    After that, apply a smaller weight decay.
    """

    def scheduler(t: Tensor) -> Tensor:
        if t < t_cool:
            return torch.tensor(wd_init, device=t.device, dtype=t.dtype)
        elif t <= t_cold:
            # apply cosine annealing
            cos_eff: Tensor = (
                (t - t_cool).mul_(torch.pi).div_(t_cold - t_cool).cos_().add_(1)
            )
            wd_start = wd_init if wd_cool is None else wd_cool
            return cos_eff.mul_(wd_start - wd_min).mul_(0.5).add_(wd_min)
        else:
            return torch.tensor(wd_min, device=t.device, dtype=t.dtype)

    return scheduler


def weight_decay_scheduler_flx(
    wd_init: float,
    wd_cool: float,
    wd_cold: float,
    t_warm: int,
    t_cool: int,
    t_cold: int,
) -> Callable[[Tensor], Tensor]:
    """
    Similar rationale to cos_anneal_lr_scheduler_flx

    t_warm: Training epoch # where weight decay rate ends being constant wd_init and start (wide) cosine annealing descent to wd_cool.
    t_cool: Training epoch # where wd descends to wd_cool and starts (narrow) cosine annealing descent to wd_cold
    t_cold: Training epoch # where wd descends to wd_cold and stays there afterwards.
    """
    assert t_warm < t_cool < t_cold

    def scheduler(t: Tensor) -> Tensor:
        if t < t_warm:
            return torch.tensor(wd_init, device=t.device, dtype=t.dtype)
        elif t <= t_cool:
            cos_eff: Tensor = (
                (t - t_warm).mul_(torch.pi).div_(t_cool - t_warm).cos_().add_(1)
            )
            return cos_eff.mul_(wd_init - wd_cool).mul_(0.5).add_(wd_cool)
        elif t <= t_cold:
            cos_eff: Tensor = (
                (t - t_cool).mul_(torch.pi).div_(t_cold - t_cool).cos_().add_(1)
            )
            return cos_eff.mul_(wd_cool - wd_cold).mul_(0.5).add_(wd_cold)
        else:
            return torch.tensor(wd_cold, device=t.device, dtype=t.dtype)

    return scheduler


def grad_clipper(
    max_norm: float,
    eps: float = 1e-6,
) -> Callable[[Iterable[nn.Parameter]], None]:
    """
    Creates logic to clip model parameters w/ given l2 norm.

    Implement this as a closure.

    max_norm: Maximum l2 norm to apply clipping.
    return: A callable which takes a list of parameters and clip their gradient
        in-place.
    """

    @torch.no_grad()
    def clip(params: Iterable[nn.Parameter]):
        """
        Spec:
        1. Compute the l2 norm of gradient of all given parameters (gn2)
        2. If gn2 < max_norm, nop and return.
        3. Else divide all parameter gradients by max_norm / (gn2 + eps) in place

        To minimize memory pressure, a naive solution may use given grad tensors
        as scratch space and restore to correct output at the end. This turns out
        introduce numerical stability issues to every gradient regardless whether
        we apply clipping or not, so avoid it at all cost.
        """
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        # NOTE logic below may lead to numerical stability issue but align w/
        # Pytorch's practice. See
        # https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/utils/clip_grad.py#L102-L108
        grad_l2_norm = linalg.vector_norm(
            torch.stack(torch._foreach_norm(grads, ord=2)), ord=2
        )

        if grad_l2_norm < max_norm:
            return

        torch._foreach_mul_(grads, max_norm / grad_l2_norm.add_(eps))

    return clip


def get_batch(
    x: npt.NDArray,
    batch_size: int,
    context_len: int,
    device: str,
    rng: np.random.Generator | None = None,
) -> tuple[
    Int64[Tensor, "batch_size context_len"], Int64[Tensor, "batch_size context_len"]
]:
    """
    x: 1-d numpy array representing a single sequence of token IDs.
    rng: RNG to produce predictable behavior for troubleshooting.

    NOTE Pytorch currently only supports *some* numpy data types. E.g.
    np.uint32 and np.uint16 are NOT supported hence a numpy array of these data
    types cannot be loaded into a Pytorch tensor via `torch.from_numpy`.

    Spec:
        Sample sequences in the resulting batch w/ sliding window.
        For i := 0, i < batch_size, do
            Turn slice x[i:i+context_len] to a tensor
            Add the tensor to list seqs
            Same applies to sequence of next tokens
        done
        Stack seqs and move the resulting tensor to given device
    """
    tokens_cnt = x.size
    assert (
        batch_size + context_len <= tokens_cnt
    ), f"Input size too small to collect {batch_size} token sequences of length {context_len}"
    in_seqs = torch.empty((batch_size, context_len), dtype=torch.int64)
    nxt_tokens = torch.empty_like(in_seqs, dtype=torch.int64)
    # sample sequences for the batch in random
    g = rng if rng is not None else _rng
    max_possible_start_idx = tokens_cnt - context_len - 1
    for ridx, i in enumerate(
        g.choice(max_possible_start_idx + 1, size=batch_size, replace=False)
    ):
        # FIXME return the same batches to verify correctness of model's math
        # for i in range(batch_size):
        # copy numpy array into a data type supported by pytorch
        in_seqs[ridx, :] = torch.from_numpy(x[i : i + context_len])
        nxt_tokens[ridx, :] = torch.from_numpy(x[i + 1 : i + 1 + context_len])

    # For speedy data transfer, move resulting batched sequence to device in async
    # More see https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    return (
        in_seqs.to(device=device, non_blocking=True),
        nxt_tokens.to(device=device, non_blocking=True),
    )


def pass_thru(
    x: npt.NDArray,
    max_batch_size: int,  # to better utilize accelerator compute
    context_len: int,
    device: str,
) -> Iterator[
    tuple[Int64[Tensor, "batch_size seq_len"], Int64[Tensor, "batch_size seq_len"]]
]:
    """
    Spec:
        Pass through data in x ASAP, in exhuastive, non-overlapping manner.
        Pass through data by max_batch_size x context_len if possible,
            Else by a smaller batch size x context_len;
            Else by 1 x length of remaining tokens.

        Use pinned memory tensor as staging buffer to transfer data
            (from disk) to GPU mem (async).

    Args:
        x: A 1-d list of tokens represented by a np array-like obj, usually a `memmap` for large dataset.
        max_batch_size: Max # token sequences to return in a batch which this function will yield.
            This means the actual # token sequences in a batch can be less than this number (e.g.
            when passing near the end of the dataset where there is inadequate data to make up a
            batch of size max_batch_size.
        context_len: Max length token sequences in a batch returned by this function can take.
            This means the actual sequence length can be less than this number.
        device: Device where the tensors to be yield by this function will be moved to.
    """
    tokens_cnt = x.size
    # Pre-allocate memory as staging buffer to transfer data from CPU to GPU
    staged_in = torch.empty(
        (max_batch_size, context_len), dtype=torch.int64, pin_memory=True
    )
    staged_nxt = torch.empty_like(staged_in, dtype=torch.long, pin_memory=True)
    # input can take token from index 0 to tokens_cnt-2,
    # nxt token can take token from index 1 to tokens_cnt-1
    idx_in_start = 0
    while idx_in_start < tokens_cnt - 1:
        batch_size, seq_len = 0, 0
        # attempt the largest stride first
        idx_in_end = idx_in_start + max_batch_size * context_len - 1
        if idx_in_end < tokens_cnt - 1:
            batch_size, seq_len = max_batch_size, context_len
        else:
            # unable to take the largest stride, so make a smaller one:
            # - A smaller batch size x context_len, OR
            # - 1 batch x length of remaining tokens
            tokens_remained = tokens_cnt - 2 - idx_in_start + 1
            batch_size, tokens_res = divmod(tokens_remained, context_len)
            if batch_size > 0:
                idx_in_end = idx_in_start + batch_size * context_len - 1
                seq_len = context_len
            else:
                idx_in_end = idx_in_start + tokens_res - 1
                # update batch size so that tensors to be yielded below has a batch dimension
                batch_size = 1
                seq_len = tokens_res

        # Load data to staging buffer. Avoid unnecessary copy and type casting
        staged_in[:batch_size, :seq_len] = torch.from_numpy(
            x[idx_in_start : idx_in_end + 1].reshape(batch_size, seq_len)
        )
        staged_nxt[:batch_size, :seq_len] = torch.from_numpy(
            x[idx_in_start + 1 : idx_in_end + 2].reshape(batch_size, seq_len)
        )
        # Then load data to target device in async
        yield (
            staged_in[:batch_size, :seq_len].to(device, non_blocking=True),
            staged_nxt[:batch_size, :seq_len].to(device, non_blocking=True),
        )
        # move starting index to that of next unvisited token
        idx_in_start = idx_in_end + 1


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    state_dict_all = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state_dict_all, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state_dict_all = torch.load(src)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict_all["model"])
    if missing_keys is not None and len(missing_keys) > 0:
        raise Exception(f"Following keys missing in model state: {missing_keys}")
    if unexpected_keys is not None and len(unexpected_keys) > 0:
        raise Exception(
            f"Following keys are not expected in model state: {unexpected_keys}"
        )

    optimizer.load_state_dict(state_dict_all["optimizer"])

    return state_dict_all["iteration"]


def gen_lr_schedule(cfg: dict) -> Callable[[Tensor], Tensor]:
    # For match-case tutorial see https://peps.python.org/pep-0636/
    match cfg:
        case {"type": "fixed", "lr": float(lr)}:
            log.info("Using fixed lr schedule: %s", cfg)
            return lambda t: torch.tensor(lr, device=t.device, dtype=t.dtype)
        case {
            "type": "cosine_annealing",
            "lr_max": float(),
            "lr_min": float(),
            "t_warm": int(),
            "t_cool": int(),
        }:
            del cfg["type"]
            log.info("Using cos annealing lr schedule: %s", cfg)
            return cosine_annealing_lr_scheduler(**cfg)
        case {
            "type": "cosine_annealing_flx",
            "lr_max": float(),
            "lr_cool": float(),
            "lr_cold": float(),
            "t_warm": int(),
            "t_cool": int(),
            "t_cold": int(),
        }:
            del cfg["type"]
            log.info("Using cos annealing flx lr schedule: %s", cfg)
            return cos_anneal_lr_scheduler_flx(**cfg)
        case _:
            raise RuntimeError(f"Unsupported learning rate schedule config {cfg}")


def gen_wd_schedule(cfg: dict) -> Callable[[Tensor], Tensor]:
    match cfg:
        case {"type": "fixed", "wd": float(wd)}:
            log.info("Using fixed weight decay rate schedule: %s", cfg)
            return lambda t: torch.tensor(wd, device=t.device, dtype=t.dtype)
        case {
            "type": "cosine_annealing",
            "wd_init": float(),
            "wd_min": float(),  # 'wd_cool': float | None = None,
            "t_cool": int(),
            "t_cold": int(),
        }:
            del cfg["type"]
            log.info("Using cos annealing weight decay rate schedule: %s", cfg)
            return weight_decay_scheduler(**cfg)
        case {
            "type": "cosine_annealing_flx",
            "wd_init": float(),
            "wd_cool": float(),
            "wd_cold": float(),
            "t_warm": int(),
            "t_cool": int(),
            "t_cold": int(),
        }:
            del cfg["type"]
            log.info("Using cos annealing flx weight decay rate schedule: %s", cfg)
            return weight_decay_scheduler_flx(**cfg)
        case _:
            raise RuntimeError(f"Unsupported weight decay rate schedule config {cfg}")


# TODO OO
def train_loop(
    train_set: npt.NDArray,
    validation_set: npt.NDArray,
    batch_size: int,
    vocab_size: int,
    context_len: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    t_max: int,
    lr_schedule_cfg: dict,
    wd_schedule_cfg: dict,
    lr: float | None,
    betas: tuple[float, float] | None,
    grad_clip_max_norm: float | None,
    tensorboard_log_dir: Path,
    from_checkpoint: Path | None = None,
    checkpoint_dir: Path | None = None,
    weight_decay: float | None = 0.01,
    checkpoint_interval: int = 1000,
    rand_seed: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    autograd_detect_anomaly: bool = False,
    eval_interval: int = 50,
    metric_interval: int = 10,
    metric_grad_norm: bool = False,
) -> None:
    """
    Args:
        lr: Initial learning rate for AdamW optimizer. This assumes we use the
            optimizer's internal scheduling instead of cosine annealling
            lr scheduler.
        t_max: Max number of training iteratins to run.
        t_warmup, t_cool: Iteration numbers marking the end of cosine annealling
            learning rate scehdule warmup phase and annealing phase.
        lr_max, lr_min: Max and min learning rate of cosine annealing schedule.
        batch_size: Batch size of training data fed to model.
        from_checkpoint: Path to the checkpointed model to load. Training run
            will resume from this checkpoint instead of starting from scratch.
        checkpoint_dir: Directory to checkpoint partially trained model. Prefer
            using hierarchical path eg ./checkpoints/my-run-with-cfg-xyz/
            Checkpoint files will have name pattern `{run_epoch_num}.pt`
        checkpoint_interval: Iteration interval to checkpoint. E.g. a value of
            7 means performing checkpointing after completing every 7 iterations.
        rand_seed: RNG seed for reproducible randomness behavior in debugging.
        autograd_detect_anomaly: Enable autograd anmaly detection. NOTE this is
            for debug only as it can slow down training progress.
        tensorboard_log_dir: Directory to output events for dashboarding w/
            Tensorboard.
        eval_interval: Iteration interval to evaluate model w/ validation dataset.
        metric_interval: Iteration interval to metric training loss.
            It shall divide eval_interval.
        metric_grad_norm: Compute and metric model parameter gradient L2 norm.
            Can slow down training. For debug only.
        weight_decay: Weight decay hyperparameter for AdamW.
        betas: AdamW hyperparams that control the updates to the moment estimates.

    TODO:
        [x] Test this w/ a very small Transformer model and try overfitting a
            single batch of training data.
        - Ability to profile CPU, RAM and accelerator usage
            - e.g. https://docs.pytorch.org/docs/stable/autograd.html#profiler
        [x] Ability to monitor and visualize training run progress
        - Different strategies to sample sequences in a batch, include but not
          limited to:
            - Fixed: Always return the same batch of sequences.
            - Given randomness: Sample w/ user-provided random seed
            - (Default) System randomness: Sample w/ randomness available on the host
    """
    train_run_args = locals()
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    with open(tensorboard_log_dir / "train.run.args.json", "w") as f:
        json.dump(train_run_args, f, default=str, indent=2)

    rng: np.random.Generator | None = None
    if rand_seed is not None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        rng = np.random.default_rng(rand_seed)

    model = TransformerModel(
        vocab_size=vocab_size,
        context_len=context_len,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=torch.device(device),
        dtype=dtype,
    )
    # optimize for efficiency when running model's tensor ops
    # See https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch-compile
    # Unfortunately as of Jan '26 Dynamo doesn't support Python3.12 :/
    # > Compilation with Inductor is not supported on mps as of torch version 2.6.0.
    # TODO knob to turn this on/off
    torch_compile_kwargs = {"backend": "inductor"}
    if device == "mps" and torch.__version__ == "2.6.0":
        torch_compile_kwargs = {"backend": "aot_eager"}
    elif device == "cuda":
        torch_compile_kwargs.update(
            {
                "mode": "max-autotune",
                #'fullgraph': True,
            }
        )
        torch._dynamo.config.verbose = True
        torch._dynamo.config.suppress_errors = False
        if dtype == torch.float32:
            torch.set_float32_matmul_precision("high")

    model.compile(**torch_compile_kwargs)

    optimizer = AdamW(
        model.parameters(),
        lr_scheduler=gen_lr_schedule(lr_schedule_cfg),
        wd_scheduler=gen_wd_schedule(wd_schedule_cfg),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    t_start = 1
    # Load checkpointed model if given
    if from_checkpoint is not None:
        t_start = load_checkpoint(from_checkpoint, model, optimizer) + 1
    # apply gradient clipping
    clip_grads = None
    if grad_clip_max_norm is not None:
        clip_grads = grad_clipper(max_norm=grad_clip_max_norm)

    summarizer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    log.info(f"Starting training run from epoch {t_start} to {t_max}")
    # TODO handle resume of training from given checkpoint
    for t in range(t_start, t_max + 1):
        should_metric = t % metric_interval == 0
        should_eval = t % eval_interval == 0
        should_checkpoint = checkpoint_dir is not None and t % checkpoint_interval == 0
        # TODO refactor code below to run_train()
        # Use 1 batch of training data per run instead of multiple batches.
        # Latter leads to unnecessary accumulation of gradient and will mess up
        # backprop.
        # > The loss will be computed over a sampled batch of data
        model.train()
        with autograd.set_detect_anomaly(autograd_detect_anomaly):
            x, targets = get_batch(train_set, batch_size, context_len, device, rng)
            # use model(...) instead of model.forward(...) as efficiency
            # optimization eg torch.compile applies to the former.
            # predictions in shape (batch_size context_len vocab_size)
            pred_logits = model(x)
            # NOTE Per cross-entropy fn math formula it accounts
            # for prediction/probability of token at each position of each sequence
            # in the batch. So passing predictions and targets tensor w/
            # sequence length dimension makes sense
            loss = cross_entropy_loss(pred_logits, targets)
            loss.backward()

            if clip_grads is not None:
                clip_grads(model.parameters())

            if metric_grad_norm and should_metric:
                with torch.no_grad():
                    grads = [p.grad for p in model.parameters() if p.grad is not None]
                    grad_l2_norm = linalg.vector_norm(
                        torch.stack(torch._foreach_norm(grads, ord=2)), ord=2
                    )
                summarizer.add_scalar(
                    "GradL2Norm", grad_l2_norm.item(), t, time.time(), new_style=True
                )

            optimizer.step()
            optimizer.zero_grad()

        # metric training loss
        if should_metric:
            summarizer.add_scalars("Loss", {"train": loss.item()}, t, time.time())

        # evaluation w/ validation dataset
        if should_eval:
            x_eval, targets_eval = get_batch(
                validation_set, batch_size, context_len, device, rng
            )
            run_eval(model, x_eval, targets_eval, summarizer, t)

        # TODO save the model upon exhausting all epochs?
        if should_checkpoint:
            time_checkpoint_start = time.time()
            checkpoint_fp = checkpoint_dir / f"{t}.pt"
            save_checkpoint(model, optimizer, t, checkpoint_fp)
            log.info(
                f"Checkpointed model at end of epoch {t:>7d}. Took {time.time()-time_checkpoint_start:>4.2f}s"
            )

    # finally compute and metric the per-token validation loss
    per_token_eval_loss(
        model,
        validation_set,
        max_batch_size=batch_size,
        context_len=context_len,
        device=device,
        summarizer=summarizer,
        step=t,
    )
    summarizer.flush()
    summarizer.close()


@torch.no_grad()
def run_eval(
    model: TransformerModel,
    x: Int64[Tensor, "batch_size context_len"],
    targets: Int64[Tensor, "batch_size context_len"],
    summarizer: SummaryWriter,
    step: int,
):
    """
    x: Batched token sequences from validation set.
    targets: Batched next tokens from validation set.
    step: Global training loop step number.
    """
    model.eval()
    pred_logits = model(x)
    loss = cross_entropy_loss(pred_logits, targets)
    # perplexity scores of sequences in batch
    ppl_batch = perplexity(pred_logits, targets)
    # Use log-perplexity for dashboarding
    lp_min, lp_median, lp_max = (
        ppl_batch.min().log_().item(),
        ppl_batch.median().log_().item(),
        ppl_batch.max().log_().item(),
    )

    now = time.time()
    summarizer.add_scalars("Loss", {"eval": loss.item()}, step, now)
    summarizer.add_scalars(
        "EvalLogPerplexity",
        {
            "min": lp_min,
            "median": lp_median,
            "max": lp_max,
        },
        step,
        now,
    )


@torch.no_grad()
def per_token_eval_loss(
    model: TransformerModel,
    validation_set: npt.NDArray,
    max_batch_size: int,
    context_len: int,
    device: str,
    summarizer: SummaryWriter,
    step: int,
):
    """Compute per-token validation loss, aka the cross-entropy loss
    averaged over all but the last tokens in the given validation dataset.

    NOTE: Expensive op if given validation set is large. Run sparsely.

    Discount the very last token because we don't know about its next token.

    When computing this metric, avoid double-counting the loss of the same
    token over and over. This thus requires an exhaustive and non-overlapping
    pass of given dataset and we cannot naively reuse get_batch.
    """
    model.eval()
    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    for batched_in, batched_targets in pass_thru(
        validation_set, max_batch_size, context_len, device
    ):
        loss.add_(
            cross_entropy_loss(model(batched_in), batched_targets).mul_(
                batched_in.numel()
            )
        )
    # compute the final result
    loss.div_(validation_set.size - 1)
    # metric the result
    now = time.time()
    summarizer.add_scalars("Loss", {"evalPerTokenAvg": loss.item()}, step, now)
