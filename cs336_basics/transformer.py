import random
import logging
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
from einops import einsum, reduce, rearrange, repeat
from cs336_basics.log import get_logger
from jaxtyping import Float, Bool, Int

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
        embedding_dim: Dimension of the embedding vectors. == d_model mentioned below
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
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=torch.float32
    ) -> None:
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
        assert d_k % 2 == 0, f"Embedding vector dimension size {d_k} is not even number"

        # (d_k/2,)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)

        # (max_seq_len,)
        positions = torch.arange(max_seq_len, device=device, dtype=dtype)

        # (max_seq_len, d_k/2)
        # freqs = torch.einsum("i,j->ij", positions, inv_freq)
        freqs = einsum(positions, inv_freq, "i,j->i j")

        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

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
        # (batch, seq, d_embed/2)
        # print(f'x shape: {x.shape} token_positions shape: {token_positions.shape}')
        cos_by_pos = self.cos[token_positions]
        sin_by_pos = self.sin[token_positions]
        # print(f'cos_by_pos shape before repeating: {cos_by_pos.shape}')

        # (batch, 1, seq, d_embed/2)
        cos_by_pos = repeat(cos_by_pos, "... pos d -> ... 1 pos d")
        sin_by_pos = repeat(sin_by_pos, "... pos d -> ... 1 pos d")
        # print(f'cos_by_pos shape: {cos_by_pos.shape}')

        # rearrange x so that the last dimension is pair of embedding vector
        # component to be rotated
        # x now of shape (batch, seq, d_embed/2, 2)
        x = rearrange(x, "... (d p) -> ... d p", p=2)
        # print(f'x shape: {x.shape}')
        # tensors below are of shape (batch, seq, d_embed/2)
        x_even_idx, x_odd_idx = x[..., 0], x[..., 1]
        # 2d rotation
        # print(f'x_even_idx shape: {x_even_idx.shape} cos_by_pos shape: {cos_by_pos.shape}')
        x_rot_even = x_even_idx * cos_by_pos - x_odd_idx * sin_by_pos
        x_rot_odd = x_even_idx * sin_by_pos + x_odd_idx * cos_by_pos
        # print(f'x_rot_even shape: {x_rot_even.shape}')

        # Re-assemble rotated pair of embedding vector elements into the whole
        # vector
        # return rearrange(
        #        torch.stack((x_rot_even, x_rot_odd), dim=-1),
        #        '... d p -> ... (d p)',
        # )
        # final = rearrange([x_rot_even, x_rot_odd], 'new ... d -> ... (d new)')
        # print(f'final result shape: {final.shape}')
        return rearrange([x_rot_even, x_rot_odd], "new ... d -> ... (d new)")


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


def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """
    TODO: doc on input and output.

    Your implementation should handle keys and queries of shape
    (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any
    number of other batch-like dimensions (if provided).
    The implementation should return an output with the
    shape (batch_size, ..., d_v).
    """
    d_k = K.shape[-1]
    # (... queries keys)
    # presoftmax = Q @ K.transpose(-1, -2) / (d_k ** 0.5)
    presoftmax = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (
        d_k**0.5
    )
    if mask is not None:
        # mask tensor serves as an index; Below `~` negates the value of mask
        # (aka True -> False and False -> True). For a slot w/ index (... q, k),
        # if the value at that index in mask tensor is False, then we assign
        # -inf to the slot w/ the same index in tensor `presoftmax`, so that
        # after exp its value becomes 0. torch.exp([-inf]) = 0
        presoftmax[..., ~mask] = -float("inf")

    # apply softmax to the very last dimension
    softmaxed = softmax(presoftmax, i=-1)
    # return softmaxed @ V
    return einsum(
        softmaxed, V, "... queries seq_len, ... seq_len d_v -> ... queries d_v"
    )


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        d_model: Dimensionality of the Transformer block inputs.
            == embedding vector dimension size d_embedding.
        num_heads: Number of heads to use in multi-head self-attention.
        theta, max_seq_len: Parameters for initializing RoPE layer.

        Folllowing Vaswani et al. [2017], set d_k = d_v = d_model / h.

        For compute efficiency, we lump W_Q, W_K, W_V into a single large
        matrix instead of separating them, to reduce # matmuls to perform.
        """
        super().__init__()
        # Initialize following linear weights per assignment section 3.4.1
        # W_Q, W_K, W_V all in shape (d_model (=h x d_k or h x d_v), d_model)
        # W_O in shape (d_model, d_model (=h x d_v))
        std = d_model**-0.5
        # Stretch goal: Combine weights for Q, K, V into a single matrix to
        # reduce # matmuls. Implement W_QKV so that Q, K, V = W_QKV[0], W_QKV[1], W_QKV[2]
        # W_QKV of shape (3, d_model, d_model)
        self.W_QKV = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(3, d_model, d_model, device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        self.W_O = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(d_model, d_model, device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )
        self.num_heads = num_heads
        if theta is not None and max_seq_len is not None:
            # For why d_k is set to size of per-head embedding dimension, see
            # the dimension of input of RoPE in forward() below
            self.rope = RotaryPositionalEmbedding(
                theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype,
            )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> torch.Tensor:
        """
        Spec:
        (NOTE we use impl notation below instead of math notation appeared in assignment handout)

        Build the attention logic bottom-up, w/ naive approach first.
            The smallest logic unit is Attention(Q_i, K_i, V_i), where i in [0, num_heads)
            Q_i = x @ (the i-th out of h slices of W_Q.T). The shape of W_Q.T is (d_model, h x d_k)

            Similar paradigm applies to K_i and V_i. Reading the original transformer paper I believe
            the dimension on which slicing applies is the dimension of embedding vector.

            (Slicing this way makes sense, cuz suppose we sliced on the dimension of token sequence,
            then when processing a token in a particular slice, how can the model attend to tokens
            which run *before* this token? No way, cuz embeddings of such predecessor tokens likely
            would have been in different slices -- a contradiction and dead end)

        Dimension along which concatenation of head_1, head_2, ... head_h is the last dim, the dimension
            of embedding vector. And the size of resulting dimension is d_model

        IMO we only need to slice right before calling scaled_dot_product_attention fn.
        Ensure correctness first before jumping to optimization.
        """
        # Reshape tensors so that on the last dimension resides per-head values subject to application of attention.
        # This way we avoid the hassle and inefficiency of for i in num_heads do Attention(Q_i, K_i, V_i) :D
        # Note d_head x d_embedding_per_head = d_embedding (aka d_model)
        # QKV_all_h of shape (... 3, d_head, d_seq, d_embeddding_per_head)
        XQKV_all_h = rearrange(
            # x @ Q.T, x @ K.T, x @ V.T in one go
            # d_to_split_by_head == (d_k or d_v x h) == d_model
            # `kind`: index subscript to select tensor for Q, k or V
            einsum(
                x,
                self.W_QKV,
                "... d_seq d_model, kind d_to_split_by_head d_model -> ... kind d_seq d_to_split_by_head",
            ),
            "... d_seq (d_head d_embedding_per_head) -> ... d_head d_seq d_embedding_per_head",
            d_head=self.num_heads,
        )
        # Apply RoPE to query and key before applying attention
        # Apply RoPE if all prerequisites present
        if hasattr(self, "rope") and token_positions is not None:
            # The 1st `:` means selecting all batches (all indices along the
            # batch dimension), the 2nd `:2` means selecting tensor for Q and K
            # More see https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
            # XQKV_all_h[:, :2] = self.rope(XQKV_all_h[:, :2], token_positions)
            # Equivalent to above, by slicing and selecting along the `kind`
            # dimension. This way can handle arbitrary prefixing dimensions in x
            XQKV_all_h[..., :2, :, :, :] = self.rope(
                XQKV_all_h[..., :2, :, :, :], token_positions
            )
        # Tensor for causal masking
        # We shall directly use the existing masking logic in scaled_dot_product_attention;
        # The problem is now reduced to creation of a mask tensor.
        # By definition, the mask should apply along the dimension of token sequence.
        # Prompting ChatGPT about the meaning of Q and K matrix, whose result of matmul over dimension (d_embedding_per_head)
        # (shape involved: (d_q x d_embedding_per_head) @ (d_embedding_per_head x d_k)) is
        # the input of masking in scaled dot product attention function, ChatGPT's response
        # claims that a row in Q, corresponding to a token in sequence, represents the "questions" / "queries"
        # which this particular token ask / attend to all other tokens in the sequence. HOWEVER,
        # I believe such interpretation should apply to the matmul result above instead; Let's say
        # the matmul result is QK (of shape (... d_q d_k)), focus on the 2-d matrix of shape (d_q d_k),
        # then the row at index i of this matrix, whose size is d_k, represents the "questions" / "queries"
        # of token i in the sequence to all other tokens in the same sequence. Item at index j of this
        # row represents the query of token i to token j in the same sequence. Meanwhile, column at index i
        # of this matrix represents the info which token at index i of sequence provides to all other tokens
        # in the same sequence. See https://chatgpt.com/s/t_6964464a2ff48191a4208206c18b4f06
        # In this light, we should create the mask so that the i-th token only
        # query all its predecessor, resulting in mask matrix whose elements of lower triangular part
        # are True while others are False
        # Note the dimension required here is the sequence length dimension which is not available in module init time :(
        d_seq_size = x.shape[-2]
        mask = torch.tril(
            torch.ones(d_seq_size, d_seq_size, device=x.device, dtype=torch.bool)
        )
        attended_all_h: Float[Tensor, "... d_head d_seq d_embedding_per_head"] = (
            scaled_dot_product_attention(
                XQKV_all_h[..., 0, :, :, :],  # XQ
                XQKV_all_h[..., 1, :, :, :],  # XK
                XQKV_all_h[..., 2, :, :, :],  # XV
                mask,
            )
        )
        # Concatenate to restore the dimension of embedding vector, so that it is ready for linear transformation w/ W_O
        attended_all_h = rearrange(
            attended_all_h,
            "... d_head d_seq d_embedding_per_head -> ... d_seq (d_head d_embedding_per_head)",
        )
        return attended_all_h @ self.W_O.T


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.norm_pre_multihead_attention = RMSNorm(d_model, eps, device, dtype)
        self.multihead_attention = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device, dtype
        )
        self.norm_pre_ffn = RMSNorm(d_model, eps, device, dtype)
        self.ffn = FFN(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """
        token_positions: Tensor holding token's position index in sequence. Can be a
        simple 1-D tensor (eg [0, 1, 2, 3, ...]) or a batched tensor.
        """
        # 1st sublayer
        x += self.multihead_attention(
            self.norm_pre_multihead_attention(x), token_positions
        )
        # 2nd sublayer
        return x + self.ffn(self.norm_pre_ffn(x))


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        vocab_size: The size of the (token/sequence item) vocabulary for
            determining dimension of token embedding matrix.
        context_length: The maximum context length, necessary for determining
            the dimensionality of the position embedding matrix.
        num_layers: Number of Transformer blocks to apply.
        d_model: Size of embedding vector dimension hidden in the Transformer model.
            Aka, the dimensionality of the model embeddings and sublayer outputs.
        """
        super().__init__()
        # layer's input and output dim: (batch, seq_len) -> (batch, seq_len, d_model)
        self.token_embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    theta=rope_theta,
                    max_seq_len=context_len,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        self.norm_post_transformer_blocks = RMSNorm(
            d_model, eps, device=device, dtype=dtype
        )
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        self.output_embedding = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        normalize_output: bool | None = True,
    ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """
        x: Batch of token ID sequences, of shape (batch_size, seq_len)
        normalize_output: Normalize transformer model output w/ softmax.
            Default to true. Add this flag as UT demands unnormalized output,
            which IMO is to avoid issue in comparison of super small numbers
            which is deemed to be inaccurate due to the use of representation
            w/ limited bitwidth.
        return: Batched normalized probability distribution over given token
            vocabulary, of shape (batch_size, seq_len, vocab_size), where the
            predicted distribution is over the next word for each input token.
            Specifically, the value at index [i, j, k] of this resultant tensor
            represents the probability that the next token of the j-th token
            in the i-th sequence of the given batch (aka x) is token w/ ID k.
            (i, j, k are all 0-based)

        For more see assignment handout section 3, page 14.
        """
        _, seq_len = x.shape
        # Tensor holding tokens position indices in sequence
        token_positions = torch.arange(0, seq_len, device=x.device, dtype=torch.int)
        x = self.token_embedding(x)
        for t in self.transformer_blocks:
            x = t(x, token_positions)
        x = self.norm_post_transformer_blocks(x)
        if normalize_output:
            return softmax(self.output_embedding(x), i=-1)
        else:
            return self.output_embedding(x)
