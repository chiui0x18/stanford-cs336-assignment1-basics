import json
from collections import Counter, defaultdict
import logging
from pathlib import Path
from heapq import heapify_max, heappop_max, heappush_max  # TODO use python 3.14
from types import new_class
from typing import DefaultDict
from .log import get_logger  # logging logic in local module
import click

log = get_logger("bpe", level=logging.WARN)

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
        self.counter: Counter[tuple[bytes, bytes]] = Counter()
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


def bpe_baseline(
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
            log.warning(
                f"Cannot merge further as token pairs have run out. Actual vocab size: {len(vocab)} Expected: {vocab_size}"
            )
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


####### Correct, functional impl of time-optimized BPE #######


class Pretoken:
    def __init__(self, seq, cnt):
        """
        seq: Token sequence representing the pretoken.
        cnt: Count of pretoken in corpus.
        """
        self.seq = seq
        self.cnt = cnt

    def __repr__(self) -> str:
        return str((self.seq, self.cnt))


def bpe_time_optimized(
    pretokens: dict[str, int], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Start: Make a full pass to all pretokens and get byte-token pair -> count
        mapping. This also yields the 1st merged token, aka the byte-token pair
        w/ the highest count. Denote this pair as p1.

    In byte-token pair -> count mapping, find pairs which are not p1 but
    overlap w/ p1; Suppose pair p meets such criteria, then it means either of
    following is true:
    - p[0] = p1[1], e.g. (b't', b'h') and (b' ', b't')
    - p[1] = p1[0], e.g. (b't', b'h') and ('h', 'e')

    NOTE such property is DIFFERENT from that p shares 1 common token w/
    p1, as the latter case implies there exist cases e.g. p[0] = p1[0] or p[1]
    = p1[1]. However we cannot merge p and p1 to create a new token -- merge is
    only possible when we can concatenate token over the overlapped element,
    e.g. merge p = (b't', b'h') and p1 = b' t' (previously merged token)
    yields pair (b' t', b'h') and token b' th'

    To determine the 2nd merged token, BPE requires us to find
    corresponding token pair w/ the highest count in text corpus; Note such
    pair may or may not overlap w/ p1.
    (as we have seen in `TinyStoriesV2-GPT4-valid.txt.pretokens.json`)

    IMO the optimization idea mentioned by assignment tries relying on the
    hypothesis that tokens created by merging a already merged token and one
    which has overlaps with it will have
    higher count compared to picking up pair of random adjacent token and count
    the pair's occurrence, hence the efficiency increase. As mentioned above,
    such hypothesis can be false, which makes us find out the pair that has
    the 2nd highest count but doesn't overlap w/ p1.
    """

    # Need a mapping from pretoken str -> token sequence which forms the
    # pretoken. Because the token sequence can change as we identify token pair
    # to merge and create new merged token from the pair.
    # To facilitate access and update of pretoken's token sequence and
    # pretoken's count, we need a "container" type, hence the Pretoken definition
    pretoken_info: dict[str, Pretoken] = {}
    for pt, cnt in pretokens.items():
        seq = tuple(bytes([b]) for b in pt.encode(UTF8))
        # BPE merge needs at least 2 tokens
        if len(seq) < 2:
            continue
        pretoken_info[pt] = Pretoken(seq, cnt)

    vocab = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    for t in special_tokens:
        # len(vocab) is the id of next new token
        new_token_id = len(vocab)
        vocab[new_token_id] = t.encode(UTF8)

    # NOTE for bpe we only concern about tokens created from merging existing ones.
    pair_cnts: Counter[tuple[bytes, bytes]] = Counter()
    # token pair -> pretokens which contain such pair
    pair_to_pretokens: DefaultDict[tuple[bytes, bytes], set[str]] = defaultdict(set)

    # iterative BPE runs
    merged_p: tuple[bytes, bytes] | None = None
    while len(vocab) < vocab_size:
        if merged_p is None:
            # Initial condition: Haven't identify any merged token
            # run a full pass of pretokens to collect initial byte-token pairs
            for pt, v in pretoken_info.items():
                # Assumption: pt contains > 1 tokens. Iterate each token and
                # collect pair of itself and its successor as merge candidate
                for p in zip(v.seq, v.seq[1:]):
                    pair_cnts[p] += v.cnt  # NOTE change in count is not 1!
                    pair_to_pretokens[p].add(pt)
        elif len(pair_cnts) == 0:
            # Unlikely to happen; But if this was true then it means there are no more
            # new merged token to be created. So log and exit loop
            log.warning(
                f"Cannot merge further as token pairs have run out. Actual vocab size: {len(vocab)} Expected: {vocab_size}"
            )
            break
        else:
            # Previous iteration has identified a merged token
            # Find candidates of the merged pair (to be identified in current iteration),
            # which can be:
            # - A pair already in pair_cnts, OR
            # - A new pair resulted from creation of merged token from previous iteration
            # and update mapping pair -> count and pair -> pretokens accordingly
            for pt in pair_to_pretokens[merged_p]:
                v = pretoken_info[pt]
                updated_token_seq(pt, v, merged_p, pair_cnts, pair_to_pretokens)
            # Drop merged_p's entry from pair_to_pretoken as it is longer needed
            pair_to_pretokens.pop(merged_p)

        # Identified following as bottleneck of BPE run performance w/ cprofile & snakeviz
        # TODO: This is to return the pair of highest count & lexical order and
        # entails heapsort (O(nlgn) where n is # pairs in pair_cnts) each time
        # it is called; Do we have a faster way than sorting every time?
        merged_p = max(
            pair_cnts, key=lambda p: PairCountSortKey(pair=p, cnt=pair_cnts[p])
        )
        merged_token = b"".join(merged_p)
        new_token_id = len(vocab)
        vocab[new_token_id] = merged_token
        merges.append(merged_p)
        log.debug(
            f"Merging pair {merged_p} of count {pair_cnts[merged_p]} to new token {new_token_id}"
        )
        debug_pair_to_pretoken_info = {
            p: [pretoken_info[pt] for pt in pts] for p, pts in pair_to_pretokens.items()
        }
        # print(f'DEBUG: pair_cnts = {pair_cnts}\npair_to_pretokens = {debug_pair_to_pretoken_info}')
        # Now remove the merged pair from pair_cnts to clear way for next merged pair
        pair_cnts.pop(merged_p)

    return vocab, merges


def updated_token_seq(
    pt: str,
    ptv: Pretoken,
    pair: tuple[bytes, bytes],
    pair_cnts: Counter[tuple[bytes, bytes]],
    pair_to_pretokens: DefaultDict[tuple[bytes, bytes], set[str]],
):
    """
    Create updated token sequence of a pretoken given
    its existing token sequence and the token pair to merge.

    This is to reflect the merge in the pretoken, the implication is that
    pair(s) which previously overlap w/ pair to merge in pretoken are now
    gone due to the merge, thus we must decrement their count accordingly

    (In this light, using a heap to manage pair counts is more awkward as
    update to items in heap are not straightforward -- one needs to find
    the item, pop it out of heap, update count then push it back. So drop
    such idea)

    To compare w/ pair(s) which has no overlap to the merged pair and find the
    next merged token (outside of this logic), save the count of *new* pairs
    resulted from merging the merged token to pair_cnts!

    pt: Pretoken string
    ptv: `Pretoken` value contain pt's token sequence and corpus count
    pair: Merged token pair
    pair_cnts: Token pair counter
    pair_to_pretokens: Mapping of pair to pretokens which contain the pair
    """
    # replace all non-overlapping occurrences of token pair w/ new token
    old = ptv.seq
    idx, ln = 0, len(old)
    new = []
    merged_token = b"".join(pair)
    # print(f'DEBUG: updating token sequence - merged pair {pair} pretoken str: "{pt}" token seq: {old}')
    while idx < ln:
        if idx < ln - 1 and old[idx] == pair[0] and old[idx + 1] == pair[1]:
            new.append(merged_token)
            # Find overlapping pairs and update their counts:
            # (old[idx-1], old[idx]) and (old[idx+1], old[idx+2])
            # Also record token(s) which can be built by merging
            # the merged token and its neighbor.
            # NOTE!!! Here pair count increments/decrements by
            # pretoken's corpus count, not 1
            if idx - 1 >= 0:
                p_gone = (old[idx - 1], old[idx])
                if p_gone in pair_cnts:
                    pair_cnts[p_gone] -= ptv.cnt
                    if pair_cnts[p_gone] == 0:
                        pair_cnts.pop(p_gone)

                new_p_w_merge_token = (old[idx - 1], merged_token)
                pair_cnts[new_p_w_merge_token] += ptv.cnt
                pair_to_pretokens[new_p_w_merge_token].add(pt)
                # print(f'DEBUG: new pair w/ merged token {new_p_w_merge_token} - merged token {merged_token}')

            if idx + 2 < ln:
                p_gone = (old[idx + 1], old[idx + 2])
                if p_gone in pair_cnts:
                    pair_cnts[p_gone] -= ptv.cnt
                    if pair_cnts[p_gone] == 0:
                        pair_cnts.pop(p_gone)

                new_p_w_merge_token = (merged_token, old[idx + 2])
                pair_cnts[new_p_w_merge_token] += ptv.cnt
                pair_to_pretokens[new_p_w_merge_token].add(pt)
                # print(f'DEBUG: new pair w/ merged token {new_p_w_merge_token} - merged token {merged_token}')

            idx += 2
        else:
            new.append(old[idx])
            idx += 1

    ptv.seq = new


class PairCountSortKey:
    """
    Sort key to find token pair of highest count and largest lexical order.

    NOTE this is a useful way to encapsulate complex comparison logic which
    cannot fit into a one-liner lambda function:

    Suppose pairs is a list of token pairs.
    Before:
    sorted(pairs, key=lambda p: # cannot fit in logic to first compare count then lexical order! ...)
    After:
    sorted(pairs, key=lambda p: PairCountSortKey(pair=p, cnt=pair_cnts[pair]))
    """

    def __init__(self, pair: tuple[bytes, bytes], cnt: int) -> None:
        self.pair = pair
        self.cnt = cnt

    def __lt__(self, other: "PairCountSortKey") -> bool:
        """
        https://docs.python.org/3/reference/datamodel.html#object.__lt__

        This pair is deemed less than other if it has a lowr count, or it is
        lexically smaller when there is a tie on count.

        This seems to work w/ max() too as long as two values in comparison
        is of same type. See https://stackoverflow.com/a/72880603
        """
        if self.cnt != other.cnt:
            return self.cnt < other.cnt
        # A tie on count; Break it by lexical ordering
        return self.pair < other.pair

    def __repr__(self) -> str:
        return str((self.pair, self.cnt))


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding=UTF8) as f:
        pretokens: dict[str, int] = json.load(f)

    return bpe_baseline(pretokens, vocab_size, special_tokens)


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
