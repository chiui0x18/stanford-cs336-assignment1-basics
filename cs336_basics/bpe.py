import json
from collections import Counter, defaultdict
import logging
from pathlib import Path
from types import new_class
from typing import DefaultDict
from .log import get_logger  # logging logic in local module
import click

log = get_logger("bpe", level=logging.DEBUG)

UTF8 = "utf8"


class BpeIterationStates:
    """
    Manage the state of a single BPE iteartion run. It does following:
        - Keeps the mapping from each token pair found to following:
            - the pair's running count, to find out the final pair to create
              new token.
            - Set of pre-token(s) where the pair is found. To efficiently
              update the mapping of pre-tokens to their count to reflect the
              presence of new token after its creation.
        - Keeps the most frequent token pair(s) found during a single BPE
          iteration run. To find the final pair to create new token.

    (IMO the pretoken -> count mapping shall be part of this state as well,
    esp. to adopt the bpe optimization idea in assignment)
    """

    def __init__(self) -> None:
        self.counter = Counter()
        self.pair_to_pretokens: DefaultDict[
            tuple[bytes, bytes], set[tuple[bytes, ...]]
        ] = defaultdict(set)
        self.max_cnt = 0
        self.most_pairs: list[tuple[bytes, bytes]] = []

    def update(
        self, pair: tuple[bytes, bytes], pretoken: tuple[bytes, ...], pretoken_cnt: int
    ):
        """
        Update the state w/ given the token pair, the pretoken where the pair is
        found, and the count of pretoken in text corpus.

        Spec:
            Increase the pair's running count by pretoken_cnt.
            Save pair -> pretoken mapping to pair_to_pretokens.
            Compare the pair's running cnt w/ self.max_cnt:
            - If cnt < self.max_cnt, nop.
            - If cnt == self.max_cnt, append pair to self.most_pairs.
            - If cnt > self.max_cnt, set self.max_cnt = cnt and set
                self.most_pairs to only contain pair.
        """
        self.counter[pair] += pretoken_cnt
        self.pair_to_pretokens[pair].add(pretoken)
        cnt = self.counter[pair]
        if cnt == self.max_cnt:
            self.most_pairs.append(pair)
        elif cnt > self.max_cnt:
            self.most_pairs = [pair]
            self.max_cnt = cnt

    def pair_to_merge(self) -> tuple[bytes, bytes] | None:
        """
        Returns pair of highest count and highest lexical order, or None if no
        pair found. Only call this after BPE iteration run finishes.
        """
        return max(self.most_pairs, default=None)


def bpe(
    pretokens: dict[str, int], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    TODO:
    - UTs

    Train a BPE tokenizer from pre-tokenized input.

    Spec:
        1. Read pre-tokenized data from input_path, which is in form of a json
            object whose key is pre-token and value is the pre-token's frequency in
            text corpus, into a dict[tuple[bytes], int]
        2. Initialize vocabulary as a dict[int, bytes], w/ int key 0 - 255 maps to
            numerically identical 1-byte bytes value. Then assign new IDs
            starting from 256 for each given special token and add
            corresponding mapping to vocabulary. Initialize list[tuple[bytes,
            bytes]] to store merges resulted from BPE.

            At this point:
                1. id for next new token = 256 + len(special_tokens).
                2. # remaining available slots in vocabulary = vocab_size - id_of_next_new_token
        3. Build up vocabulary by running BPE till vocabular size hits
            vocab_size iteratively. Start iteration from pairing adjacent byte-level
            tokens present in each pre token (indiviaul key from step 1). Use
            a Counter to manage the running frequency count of each pair found. Each
            iteration is to find the token pair w/ the highest frequency, and
            when there are multiple pairs w/ the same highest frequency, break
            tie by choosing the pair w/ highest lexico order. Specifically:
                For each pre-token pt:
                    Start from byte index 0, pick pair of adjacent byte-level
                    tokens and increment its count in Counter.
                    Need a way to record the mapping from a pair to pre-tokens
                    which contain such pair, so that later on we can find these
                    pre-tokens in O(1) instead of blindly visiting all
                    pre-tokens. Need a mapping dict[bytes, set[pre-tokens that
                    contains the pair]]

            For efficiency, Counter doesn't
            have an eas way to get keys whose count is the highest so will need
            new logic to track those keys if we want to avoid repeatedly
            sorting Counter's entries to find them out. Must to track all pairs
            w/ highest frequency as future pairs of highest frequency are built
            on top of them.

            Once we find the pair of highest freq and break the tie, give it a
            new ID and add it to vocabulary, also add the pair to merges list.

    """
    # tuple[bytes] -> int
    token_seq_cnts = {
        tuple(bytes([b]) for b in pt.encode(UTF8)): cnt for pt, cnt in pretokens.items()
    }
    vocab = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    for t in special_tokens:
        # len(vocab) is the id of next new token
        new_token_id = len(vocab)
        vocab[new_token_id] = t.encode(UTF8)

    # NOTE for bpe we only concern about tokens created from merging existing
    # ones.
    while len(vocab) < vocab_size:
        state = BpeIterationStates()
        for tokens, pretoken_cnt in token_seq_cnts.items():
            # Ignore 1-byte pretoken
            if len(tokens) == 1:
                continue
            # Current pretoken contain > 1 tokens. Iterate each token and
            # collect pair of itself and its successor as merge candidate
            for p in zip(tokens, tokens[1:]):
                state.update(p, tokens, pretoken_cnt)

        p_to_merge = state.pair_to_merge()
        if p_to_merge is None:
            log.warn("Pair to merge is not found. This shall not happen!")
            break
        new_token = b"".join(p_to_merge)
        # NOTE check whether the new token already existed in vocab?
        new_token_id = len(vocab)
        vocab[new_token_id] = new_token
        merges.append(p_to_merge)
        log.debug(
            f"Merging pair {p_to_merge} of count {state.counter[p_to_merge]} to new token {new_token_id}"
        )

        # update token_seq_cnts to reflect the presence of new token
        pretokens_to_update = state.pair_to_pretokens[p_to_merge]
        for pretoken in pretokens_to_update:
            # keep its count in token_seq_cnts
            cnt = token_seq_cnts.pop(pretoken)
            # replace all non-overlapping occurrences of token pair in pretoken w/ new token
            idx, ln, updated_pretoken = 0, len(pretoken), []
            while idx < ln:
                if (
                    idx < ln - 1
                    and pretoken[idx] == p_to_merge[0]
                    and pretoken[idx + 1] == p_to_merge[1]
                ):
                    updated_pretoken.append(new_token)
                    idx += 2
                else:
                    updated_pretoken.append(pretoken[idx])
                    idx += 1
            # save mapping updated_pretoken -> cnt back to token_seq_cnts
            token_seq_cnts[tuple(updated_pretoken)] = cnt

    return vocab, merges


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding=UTF8) as f:
        pretokens: dict[str, int] = json.load(f)

    return bpe(pretokens, vocab_size, special_tokens)


@click.command(name="bpe")
@click.argument("infile", type=click.Path(exists=True, path_type=Path))
@click.argument("size", type=click.INT)
def run(infile: Path, size: int):
    """
    INFILE the file path of mapping of pretokens to their counts in text
    corpus.

    SIZE the size of vocabulary that byte-pair encoding logic will build.
    """
    train_bpe(
        str(infile),
        vocab_size=size,
        special_tokens=["<|endoftext|>"],
    )


if __name__ == "__main__":
    run()
