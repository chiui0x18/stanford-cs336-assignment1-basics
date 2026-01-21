from math import sqrt
from collections.abc import Iterable
from typing import Any, Callable, Optional
import torch
from torch.optim import Optimizer
from torch import Tensor
from cs336_basics.log import get_logger
from jaxtyping import Float, Bool, Int


def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"],
) -> Float[Tensor, ""]:
    """
    input: Logits predicted by model. input[i] are the unormalized scores representing
        the probability distribution of next item for the i-th sequence in given batch.
    targets: The true next token of each sequence in given batch, serving as ground truth in computing the loss.
    return: Cross entroy loss value across all sequences in given batch.

    Spec:
        For numerical stability of softmax, identify the max value per row in input and subtract input by these values, resulting tensor input'.
        Compute softmax of input' over the last dim.
        Get the predicted probability of ground truth token per sequence from input' w/ advanced indexing over targets, denoting the resultant tensor predicted.
        Return predicted's avg as final result.

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
    # shape (batch_size )
    logsumexp = torch.logsumexp(inputs, dim=-1)
    # shape (batch_size 1) so squeeze out the last dim for subsequent subtraction
    ground_truth_item_logits = torch.gather(
        inputs, dim=-1, index=targets.unsqueeze(dim=-1)
    ).squeeze(dim=-1)
    return (logsumexp - ground_truth_item_logits).mean()


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        '''
        Below presents a naive AdamW implementation.
        It is correct, however neither Pytorch idiomatic nor performant.

        Per ChatGPT, hardware accelerators excel when all math ops are expressed as tensor operations.

        TODO: Refactor for performance and being more idiomatic.
        '''
        loss = None if closure is None else closure()
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
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                lr_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
                d1 = lr_t * m / (torch.sqrt(v) + eps)
                d2 = lr * weight_decay * p.data
                p.data -= d1 + d2

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
