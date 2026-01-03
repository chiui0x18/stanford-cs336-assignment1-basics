import torch
import torch.nn as nn
from einops import einsum

# TODO https://docs.pytorch.org/docs/stable/notes/randomness.html
RAND_SEED = 10241


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.w = nn.Parameter(
            nn.init.trunc_normal_(
                # order dimensions in row-major order for compute efficiency.
                # Cuz the c/c++ lib used by Pytorch uses mem storage
                # representation of row-order, access logic written at higher
                # level (Python code) needs to align to avoid negative impact
                # to performance. See https://pytorch.org/blog/tensor-memory-format-matters/
                torch.empty(out_features, in_features, device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        print(self.w.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In vanilla torch:
        # return x @ self.w.T
        # Prefer Einstein notation as it clearly and concisely declares the op's purpose
        # NOTE use of ein notation can entail negative impact to performance;
        # But the crucial idea is to achieve correctness and functionality before
        # even worry about performance / optimization. So stay disciplined.
        return einsum(x, self.w, "... d_in , d_out d_in -> ... d_out")
