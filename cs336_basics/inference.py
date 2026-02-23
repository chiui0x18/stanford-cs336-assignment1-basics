import torch
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import softmax, TransformerModel
from cs336_basics.log import get_logger

log = get_logger("inference")

class Predictor:

    def __init__(
        self, tokenizer:
        Tokenizer, model: TransformerModel,
        temperature: float = 0.5,
        sampling_mode: str | None = None,
        p: int | None = None,
    ) -> None:
        '''
        sampling_mode: How to sample/pick the next token from model's
            predicted next-token probability distribution. Possible values:
            - None: Pick token w/ highest probability
            - 'nucleus': Nucleus / top-p sampling. This mode uses Pytorch
                default RNG for randomness.
        temperature: Temperature scaling parameter. Smaller value means
            more contrast b/w token w/ max probability and others.
        '''
        self.tokenizer = tokenizer
        self.model = model
        assert temperature > 0, 'Temperature scaling parameter must be positive'
        self.temperature = temperature

        match sampling_mode:
            case None:
                self.sampling_mode = None
            case 'nucleus':
                assert p is not None and p > 0, 'Nucleus sampling threshold must be positive'
                self.sampling_mode = sampling_mode
                self.p = p
            case _:
                raise RuntimeError(f'Unsupported sampling mode: {sampling_mode}')

        self._initiated = False

    def _init(self) -> None:
        """Lazy initialization."""
        if self._initiated:
            return
        # get device used by the model to initiate context window mem buffer
        self.device = self.model.device
        # Token IDs ring buffer
        self.ctx_buf: list[int] = [-1] * self.model.context_len
        # Indices to manage ring buffer state
        self.buf_idx_s, self.buf_idx_e = -1, -1
        self._initiated = True

    def _buffer(self, tokens: list[int]):
        '''Add given tokens to context window staging buffer. Assume tokens can fit in.

        Move buf end ptr to fit prompt's tokens into ctx.
        If accumulated data overflows buf:
            Overflow happens if end ptr oversteps start ptr *after* it moves.
            Discard oldest data by moving buf start ptr so that it is after end ptr by 1.
            
        Init/Empty buf: s == e < 0
        Has data. Data size < buf size: s < e
        Overflow: New data comes in, move e to new pos to store new data and find it oversteps s (aka e == s)
        '''
        buf, s, e = self.ctx_buf, self.buf_idx_s, self.buf_idx_e
        buf_len = len(buf)
        for t in tokens:
            # find slot to store the token
            if e < 0:
                # Init: No data in buf
                s = e = 0
            else:
                # Data exists in buf and e indexes existent data. Find a new slot by moving e
                e = (e+1) % buf_len
                # check if the move results in overflow
                if e == s:
                    # TODO handle overflow by dropping the oldest data
                    s = (s+1) % buf_len
            # save token to buf
            buf[e] = t
        # update buf state stored in instance for future ops
        self.buf_idx_s, self.buf_idx_e = s, e

    def _buffered_tokens(self) -> list[int]:
        '''Return all tokens in the context window staging buffer in FIFO order.'''
        buf, s, e = self.ctx_buf, self.buf_idx_s, self.buf_idx_e
        if s <= e:
            return buf[s:e+1]
        # s > e: Wrap-around case
        return buf[s:] + buf[:e+1]

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        out_tokens_limit: int,
        eot_token: str | None = None,
        rewind_ctx: bool = False,
    ) -> str:
        """Generate predicted text tokens given prompt.

        Spec:
        Convert prompt to tokens w/ tokenizer.
        Move prompt's tokens into context window staging buffer to transfer the
        data to model's device.
        Get next-token probability distribution by predicting w/ model and
            existent tokens in buffer. Then sample the next token from this
            distribution.

        Check whether to end generation or not:
            - Is the predicted token an end-of-text token?
            - Is # predicted tokens hitting given limit?
            Stop generation if any of above is true.
            Then decode all predicted tokens and return the resultant string.

        If continue generation:
            Add the predicted token to context window staging buffer.
            Repeat the data transfer and prediction above.
        """
        self._init()
        if rewind_ctx:
            self.buf_idx_s = self.buf_idx_e = 0

        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_ln = len(prompt_tokens)
        ctx_len = self.model.context_len
        assert (
            prompt_ln <= ctx_len
        ), f"Given prompt cannot fit into context window: {prompt_ln} > {ctx_len} tokens"

        eot_token_id = None
        if eot_token is not None:
            tokens = self.tokenizer.encode(eot_token)
            assert len(tokens) == 1
            eot_token_id = tokens[0]

        self._buffer(prompt_tokens)
        # track # tokens generated
        tokens_out: list[int] = []

        while len(tokens_out) < out_tokens_limit:
            # shape (1 seq_len)
            t_in = torch.tensor([self._buffered_tokens()], device=self.device, dtype=torch.int64)
            # shape (1 seq_len vocab_size)
            logits = self.model(t_in)
            idx = self.sample_next_token(logits)
            # Break if we hit end-of-text token
            if idx == eot_token_id:
                break
            idx_ = idx.item()
            self._buffer([idx_])
            tokens_out.append(idx_)

        return self.tokenizer.decode(tokens_out)

    def sample_next_token(self, logits: Float[Tensor, '1 seq_len vocab_size']) -> Int[Tensor, '']:
        if self.sampling_mode is None:
            # Naive sampling by picking the id of token w/ highest unormalized probability
            return logits[0, -1, :].argmax()

        # Apply Nucleus sampling.
        # First Apply temperature scaling. Output shape (vocab_size,)
        logits = logits[0, -1, :] / self.temperature
        # Tokens w/ higher interpreted probability now come first
        logits_sorted, token_ids_sorted = torch.sort(logits, descending=True)
        prob_cumsum = torch.softmax(logits_sorted, dim=-1).cumsum(dim=-1)
        over_threshold = prob_cumsum > self.p
        # Keep the 1st token which let accumulated sum go over threshold by flipping its value to false
        # NOTE we achieve this w/ softmax and cumsum's monotonic nature, no need to know the
        # location of such token in the tensor.
        over_threshold[1:] = over_threshold[:-1].clone()
        # Chances are that a very low value of p can result in dropping of all tokens in vocab,
        # resulting in CUDA err warning presence of inf. So always keep one w/ the highest interpreted probability.
        over_threshold[0] = 0
        # Get indices of tokens to drop, aka those whose over_threshold[idx] = True
        # Remaining tokens form the "nucleus" mentioned in paper
        token_ids_drop = token_ids_sorted[over_threshold]
        # Set the logits of tokens to drop to -inf so that the interpreted probability
        # of picking them is 0 after softmax
        logits_sorted[token_ids_drop] = -float('inf')
        # Re-compute softmax then randomly sample 1 out of the remaining tokens.
        # NOTE because we have set the probability of tokens to drop to 0,
        # multinomial sampling will therefore exclude them.
        return torch.multinomial(torch.softmax(logits_sorted, dim=-1), 1)