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
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> None:
        """
        lr: learning rate
        weight_decay: weight decay rate
        """
        defaults = {
            "lr": torch.tensor(lr, device=device, dtype=dtype),
            "beta1": torch.tensor(betas[0], device=device, dtype=dtype),
            "beta2": torch.tensor(betas[1], device=device, dtype=dtype),
            "eps": torch.tensor(eps, device=device, dtype=dtype),
            "weight_decay": torch.tensor(weight_decay, device=device,
                                         dtype=dtype),
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
                    state['t'] = torch.tensor(1, dtype=p.dtype)
                    state['m'] = torch.zeros_like(p,
                                                  memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p,
                                                  memory_format=torch.preserve_format)

                t: torch.Tensor = state.get("t")
                m: torch.Tensor = state.get("m")
                v: torch.Tensor = state.get("v")

                # paradigm commented out below is discouraged in Pytorch
                # grad = p.grad.data
                with torch.no_grad():
                    grad = p.grad
                    # use in-place tenor ops to avoid unnecessary mem pressure
                    # due to creation of intermediate tensors
                    m.mul_(beta1).add_(grad, alpha=1-beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    # Apply weight decay first; See spec in Pytorch doc
                    p.mul_(1 - lr * weight_decay)
                    lr_t = lr * torch.sqrt(1 - beta2**t) / (1 - beta1**t)
                    # descent
                    p.addcdiv_(m, torch.sqrt(v).add_(eps), value=-lr_t)

                t.add_(1)

        return loss
