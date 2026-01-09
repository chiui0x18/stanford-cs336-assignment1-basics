from math import sin, cos
import random
import logging
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from einops import einsum, reduce, rearrange
from cs336_basics.log import get_logger
from jaxtyping import Float

# https://docs.pytorch.org/docs/stable/notes/randomness.html
# NOTE this has negative performance implications so be aware when using it in
# production
RNG_SEED = 10241
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

log = get_logger("transformer", level=logging.DEBUG)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        in_features: Final dimension of the input
        out_features: Final dimension of the output
        device: Device to store model parameters on
        dtype: Model parameter datatype
        """
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        self.w = nn.Parameter(
            nn.init.trunc_normal_(
                # Lay dimensions in row-major order for performant execution.
                # Cuz the c/c++ lib used by Pytorch uses mem storage
                # representation of row-order, access logic written at higher
                # level (Python code) has to align to avoid negative impact
                # to performance. See https://pytorch.org/blog/tensor-memory-format-matters/
                torch.empty(out_features, in_features, device=device, dtype=dtype),
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


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        num_embeddings: Vocabulary size
        embedding_dim: Dimension of the embedding vectors
        device: Device to store model parameters on
        dtype: Model parameter datatype

        See https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        """
        super().__init__()
        # Mapping of token (identified by its id) -> The vector representing
        # the token in embedding space. For token whose id = i, its embedding
        # vector is the tensor of dimension=1 and shape=(embedding_dim,) at
        # token_to_embedding_vector[i]
        self.token_to_embedding_vector = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: A tensor of long type of shape (batch, seq_len)
        return: A tensor of token embeddings of shape (batch, seq_len, embedding_dim)

        Spec:
        Embedding layer serves as a mapping from individual token (identified
        by its id) to corresponding embedding vector (of size `embedding_dim`)

        So looking up embedding vector by given token id, assemble the found
        vectors into a matrix then return is what embedding layer needs to do.

        NOTE although embedding layer serves only as lookup table to map token
        to vector in embedding space as part of the model's forward pass
        computation, the parameters of embedding layer DO participate in
        backpropagation and have to be updated per gradient descent. Therefore
        embedding layer parameters are learned from LLM training process
        instead of pre-trained and fixed up-front.

        The resulting matrix will be of shape (batch, seq_len, embedding_dim)

        To construct this matrix in imperative, non-linear algebra manner:
            For each seq in batch:
                For each token in seq:
                    Get vector at self.embedding[token's ID] and add it to
                        resulting matrix

        The goal is to do this in pytorch-idiomatic, linear algebra way so that
        it can be done in parallel on accelerator eg GPU. The sequential
        implementation paradigm made as-it-is from the pseudocode above is not
        suitable here so we have to ditch it.

        Pytorch achieves this w/ its advanced indexing mechanism, which very
        likely influenced by Numpy's indexing paradigm. See:
        - https://numpy.org/doc/stable/user/basics.indexing.html
        - https://numpy.org/neps/nep-0021-advanced-indexing.html
        """
        # NOTE this returns a new copy
        return self.token_to_embedding_vector[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square normalization block."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
    ) -> None:
        """
        d_model: Hidden dimension of the model, aka embedding vector size
        eps: HYPERPARAM Epsilon value for numerical stability
        """
        super().__init__()
        self.gains = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: in shape (batch_size, sequence_length, d_model)
        return: A tensor of shape (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        # Q: shall we fix the dtype of model params (gains) to float32?
        x = x.to(torch.float32)
        d_model_size = x.shape[-1]
        # Reduce along x's final dimension of size d_model in square sum
        rms = torch.sqrt(
            reduce(torch.square(x), "b s d_model -> b s", reduction="sum")
            / d_model_size
            + self.eps
        )
        # For element-wise division to work we need to satisfy the broadcasting semantics
        # Here rms is in shape (b, s) while the dividend (number to be divided) in shape (b, s, d_model)
        # Thus adjust rms to shape (b, s, 1) for broadcasting to work
        rms = rearrange(rms, "b s -> b s 1")
        # print(f'rms in shape {rms.shape}: {rms}')
        # print(f'prod in shape {prod.shape}: {prod}')
        res = x * self.gains / rms
        return res.to(in_dtype)


class FFN(nn.Module):
    """Position-Wise Feed-Forward Network"""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            # set d_ff = 8/3 x d_model to minimize loss increase
            d_ff = 8 * d_model // 3
            log.debug("Computed feed forward network dimension size: %d", d_ff)
        self.w1 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )
        self.w3 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_ff, d_model, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )
        self.w2 = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_ff, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spec:

        Different from math notation which is column-major, here we apply matrix operation in row-major manner.
        """
        # x_w1 = x @ self.w1.T
        x_w1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        x_w1_sigmoid = x_w1 * torch.sigmoid(x_w1)
        # x_w3 = x @ self.w3.T
        x_w3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        # (x_w1_sigmoid * x_w3) @ self.w2.T
        return einsum(
            (x_w1_sigmoid * x_w3), self.w2, "... d_ff, d_model d_ff -> ... d_model"
        )


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        """
        This module is used in transformer model component "Causal Multi-Head Self-Attention w/ RoPE"
        Its input is the output of pre-norm layer, in shape (batch_size, seq_len, d_embedding)

        This is how we model positions in a transformer, aka encoding position-specific
        info into the transformer model.
        See https://youtu.be/ptFiH_bHnJw?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_&t=2007

        This is to construct a mathematical tool to measure how "far" a pair of tokens
        are away from each other, aka *relative* position b/w a pair of token.

        The idea is to use the observation that *inner product of two vector is invariant
        to arbitrary rotation* to represent our desired characteristic that token embedding
        should be invariant to absolute position.

        d_k: dimension of query and key vectors, = d_embedding
        (Note it seems we assume d_k to be a even number)

        max_seq_len: Maximum sequence length. Used only for caching pre-computed rotation
        angle values. Not necessarily = seq_len

        spec:
        i: token position index (in a token sequence). IMO i in [0, max_seq_len)
            Q: Does this index start from 0 or 1? Assignment handout doesn't tell. But
            per my experience in math notation indexing starting from 1 seems prevailing
            A: Per UT, i starts from 0 (UT will barf if we use 1-based index)

        R: Matrix that contain pair-wise rotation matrix R_i, where i in [0, max_seq_len)
            R_i is of shape (d_k, d_k), so R is of shape (max_seq_len, d_k, d_k) to
            account for ALL token positions in a token sequence.

        The constructor logic is to compute and cache R.

        If we did this in sequential manner:
        For i in [0, max_seq_len):
            Compute R_i.
            For k in [1, d_k/2]:
                Compute rotation angle theta_i_k
                    theta_i_k = (i+1) / (theta^((2k-2)/d_k))
                Compute cos and sin value at theta_i_k, form 2x2 matrix block R_i_k
                Stack the block diagonally into R_i
            R[i] = R_i

        TODO: More pytorch-idiomatic, parallelizable impl
        """
        super().__init__()
        assert d_k % 2 == 0, f"{d_k} is not even number"

        def R_i_block(i: int, k: int) -> torch.Tensor:
            theta_i_k = i / (theta ** ((2 * k - 2) / d_k))
            cos_theta_i_k, sin_theta_i_k = cos(theta_i_k), sin(theta_i_k)
            return torch.tensor(
                [
                    [cos_theta_i_k, -sin_theta_i_k],
                    [sin_theta_i_k, cos_theta_i_k],
                ],
                device=device,
            )

        # R
        self.register_buffer(
            "rotation_matrix",
            # == torch.stack(tensors, dim=0)
            rearrange(
                [
                    # R_i
                    torch.block_diag(*[R_i_block(i, k) for k in range(1, d_k // 2 + 1)])
                    for i in range(0, max_seq_len)
                ],
                # stack all R_i together along the new (1st) dimension
                "... -> ...",
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (..., seq_len, d_k)
        token_positions: a tensor of shape (..., seq_len) specifying the token
            positions of x along the sequence dimension.
        return: a tensor of the same shape as x.

        Spec:
        Identify the rotation matrices to use by given token position.
            Do this w/ Pytorch's advanced indexing: R[token_positions] -> tensor of shape (..., seq_len, d_k, d_k)
        Multiply a token's embedding vector of shape (d_k,) w/ the corresponding identified rotation matrix of shape (d_k, d_k)
        """
        rotation_matrix_by_pos: Float[Tensor, "... d_k d_k"] = self.rotation_matrix[
            token_positions
        ]
        # note in einsum notation we cannot use duplicated
        return einsum(
            x,
            rotation_matrix_by_pos,
            # NOTE again in implementation we have to use row-major order,
            # aka token_embedding_vector @ R_i.T Otherwise the result will
            # be incorrect.
            "... d_k_in, ... d_k_out d_k_in -> ... d_k_out",
        )


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    """
    x: Multi-dimensional tensor.
    i: The i-th dimension of x (For now assume the index is 0-based)
    return: Tensor of same shape as x, w/ values at i-th dimension turn into
        normalized probabilities under softmax.

    Can refer to the behavior of std pytorch lib softmax function.
    """
    assert (
        -x.ndim <= i < x.ndim
    ), f"Given dimension index {i} not found in tensor of {x.ndim} dimensions"
    # Get the max values and corresponding indices on dimension i, plus keeping
    # x's dimension around. This saves the extra move of creating a new
    # dimensin in the resulting tensor (eg via tensor.unsqueeze(dim=i))
    max_on_dim_i = x.max(dim=i, keepdim=True)
    # softmax formula
    # Subtract values on dimension i by the max values so that exp()
    # of these values yields small values in (0, 1] for numerical stability
    exp = torch.exp(x - max_on_dim_i.values)
    exp_sum_on_dim_i = torch.sum(exp, dim=i, keepdim=True)
    return exp / exp_sum_on_dim_i
