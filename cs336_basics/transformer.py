import torch
import torch.nn as nn
from einops import einsum

# TODO https://docs.pytorch.org/docs/stable/notes/randomness.html
RNG_SEED = 10241


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
        torch.manual_seed(RNG_SEED)
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
        torch.manual_seed(RNG_SEED)
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

        Spec:
        Embedding layer serves as a mapping from individual token (identified
        by its id) to corresponding embedding vector (of size `embedding_dim`)

        So looking up embedding vector by given token id, assemble the found
        vectors into a matrix then return is what embedding layer needs to do.

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
