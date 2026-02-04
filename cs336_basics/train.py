import sys
import time
import random
from pathlib import Path
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable, Callable
from typing import BinaryIO, IO
import os
import torch
from torch.optim import Optimizer
from torch import Tensor, nn, linalg, autograd
from torch.utils.tensorboard.writer import SummaryWriter
from jaxtyping import Float, Int, Int64

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
    ) -> None:
        """
        lr: learning rate
        weight_decay: weight decay rate
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
                    state["t"] = torch.tensor(1, dtype=p.dtype)
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
                    # Apply weight decay first; See spec in Pytorch doc
                    p.mul_(1 - lr * weight_decay)
                    # Compute lr w/ scheduler if present
                    if self.lr_scheduler is not None:
                        lr_t = self.lr_scheduler(t)
                    else:
                        lr_t = lr * torch.sqrt(1 - beta2**t) / (1 - beta1**t)
                    # descent
                    p.addcdiv_(m, torch.sqrt(v).add_(eps), value=-lr_t)

                t.add_(1)

        return loss


def cosine_annealing_lr_scheduler(
    lr_max: float,
    lr_min: float,
    t_warmup: int,
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
                        lr_max=1e-1, lr_min=1e-3, t_warmup=100, t_cool=900)

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
        if t < t_warmup:
            return (t / t_warmup).mul_(lr_max)
        elif t <= t_cool:
            cos_eff: Tensor = (
                (t - t_warmup).mul_(torch.pi).div_(t_cool - t_warmup).cos_().add_(1)
            )
            return cos_eff.mul_(lr_max - lr_min).mul_(0.5).add_(lr_min)
        else:
            return torch.tensor(lr_min, device=t.device, dtype=t.dtype)

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
    in_seqs, nxt_tokens = [], []
    # sample sequences for the batch in random
    g = rng if rng is not None else _rng
    max_possible_start_idx = tokens_cnt - context_len - 1
    for i in g.choice(max_possible_start_idx + 1, size=batch_size, replace=False):
        # FIXME return the same batches to verify correctness of model's math
        # for i in range(batch_size):
        # copy numpy array into a data type supported by pytorch
        in_seqs.append(torch.from_numpy(x[i : i + context_len].astype(np.int64)))
        nxt_tokens.append(
            torch.from_numpy(x[i + 1 : i + 1 + context_len].astype(np.int64))
        )

    return (
        torch.stack(in_seqs, dim=0).to(device=device),
        torch.stack(nxt_tokens, dim=0).to(device=device),
    )


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
    lr: float | None,
    t_max: int,
    t_warmup: int | None,
    t_cool: int | None,
    lr_max: float | None,
    lr_min: float | None,
    grad_clip_max_norm: float | None,
    from_checkpoint: Path | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 1000,
    rand_seed: int | None = None,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    autograd_detect_anomaly: bool = False,
    tensorboard_log_dir: Path | None = None,
    eval_interval: int = 50,
    metric_interval: int = 10,
    metric_grad_norm: bool = False,
    weight_decay: float | None = 0.01,
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
        TODO: Expose betas as well.

    TODO:
        [x] Test this w/ a very small Transformer model and try overfitting a
            single batch of training data.
        - Ability to profile CPU, RAM and accelerator usage
            - e.g. https://docs.pytorch.org/docs/stable/autograd.html#profiler
        - Ability to monitor and visualize training run progress
        - Different strategies to sample sequences in a batch, include but not
          limited to:
            - Fixed: Always return the same batch of sequences.
            - Given randomness: Sample w/ user-provided random seed
            - (Default) System randomness: Sample w/ randomness available on the host
    """
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
    torch_compiler_backend = "inductor"
    if device == "mps" and torch.__version__ == "2.6.0":
        torch_compiler_backend = "aot_eager"
    model.compile(backend=torch_compiler_backend)

    lr_scheduler = cosine_annealing_lr_scheduler(
        lr_max=lr_max, lr_min=lr_min, t_warmup=t_warmup, t_cool=t_cool
    )
    optimizer = AdamW(model.parameters(),
                      lr=lr_max, lr_scheduler=lr_scheduler,
                      weight_decay=weight_decay)

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
            predictions = model(x, normalize_output=False)
            # NOTE Per cross-entropy fn math formula it accounts
            # for prediction/probability of token at each position of each sequence
            # in the batch. So passing predictions and targets tensor w/
            # sequence length dimension makes sense
            loss = cross_entropy_loss(predictions, targets)
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
    preds = model(x, normalize_output=False)
    loss = cross_entropy_loss(preds, targets)

    # perplexity scores of sequences in batch
    ppl_batch = perplexity(preds, targets)
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
