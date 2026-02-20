from collections.abc import Iterator
import torch
from torch import Tensor
from jaxtyping import Int
from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import TransformerModel


class Predictor:

    def __init__(
        self, tokenizer: Tokenizer, model: TransformerModel, ctx_len: int
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.ctx_len = ctx_len
        self._initiated = False

    def _init(self) -> None:
        """Lazy initialization."""
        if self._initiated:
            return
        # get device used by the model to initiate context window mem buffer
        self.device = next(self.model.parameters()).get_device()
        # CPU tensor as ctx window staging buffer
        self.ctx_buf = torch.empty((self.ctx_len,), dtype=torch.int64)
        # index tracking slot to save predicted next token in staging buffer
        self.nxt_token_idx = 0
        self._initiated = True

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        out_tokens_limit: int,
        eot_token: str,
        rewind_ctx: bool = False,
    ) -> Iterator[str]:
        """Generate predicted text tokens given prompt.

        Spec:
        Convert prompt to tokens w/ tokenizer.
        Move prompt's tokens into context window staging buffer to transfer the
        data to model's device.
        Predict the next token given existing ones w/ model, getting raw
        logits.
        Identify the next token w/ tokenizer, record it in buffer for output.
        Check whether to end generation or not:
            - Is the predicted token an end-of-text token?
            - Is # predicted tokens hitting given limit?
            Stop generation if any of above is true.
            Then decode all predicted tokens and return the resulting string.

        If continue generation:
            Check whether we max out ctx window or not.
                If so, make room for the predicted token in ctx window staging
                buffer, by moving buf[1:buf_size] to buf[0:buf_size-1]. Save
                the predicted token at buf[-1]

                Otherwise, save the predicted token to the next available slot
                in ctx window staging buf.

            Repeat the data transfer and prediction above.
        """
        self._init()
        raise NotImplementedError
