import json
from collections import Counter, defaultdict
import logging
from pathlib import Path
from typing import DefaultDict
from heapq import heapify_max, heappush_max, heappop_max  # requires python3.14
from .log import get_logger  # logging logic in local module

# Comment out imports like such which not needed by UTs to speed things up
# TODO move the CLI logic to a different py module instead?
# import click

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
    Functional but inefficient BPE algo serving as baseline.

    Spec (maybe outdated, use code as authority when in doubt):
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


def bpe_optimized(
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

    Idea to remove the bottleneck: Use a max heap to track the next merged
    pair. Item in the heap will be of PairCount type, and we will use
    PairCount.valid = False to signal that an item is no longer valid because
    the corresponding pair's count has changed. To preserve the correctness of
    loose ordering b/w `PairCount` instances in the max heap, which allows us
    to correctly track the next merged pair, we avoid updating the value of
    `cnt` attribute of PairCount instances whenever we observe the count of a
    token pair gets updated due to merge of token pair selected by BPE run.
    Instead we set PairCount instance's `valid` attr to False upon change in
    its count so that we are able to examine whether a PairCount instance we
    pop from the max heap is valid or not (and keep popping until we find a
    valid one or till heap is empty).
    NOTE this is not very ideal (aka tracking the next merged pair which has
    highest count and lexical order in O(1) time), but at least we no
    longer need to sort all the known pair-cnt entries (O(nlgn)) - we will only
    need 2 x |pairs w/ updated count| + |new pairs| heap push/pop operations,
    (where 2x is due to popping invalid pairs and pushing updated pairs)
    resulting time complexity of O((2 x |pairs w/ updated count| + |new pairs|)lgn)
    NOTE such seemingly less-ideal solution turned out to provide us significant
    speedup in BPE iteration process.
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

    pair_cnts: dict[tuple[bytes, bytes], PairCount] = {}
    # token pair -> pretokens which contain such pair
    pair_to_pretokens: DefaultDict[tuple[bytes, bytes], set[str]] = defaultdict(set)
    # PairCount max heap
    pch: list[PairCount] = []

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
                    pc = pair_cnts.get(p, PairCount(p, 0))
                    pc.cnt += v.cnt  # NOTE change in count is not 1!
                    pair_cnts[p] = pc
                    pair_to_pretokens[p].add(pt)
            # Build a max heap from created PairCount values, in linear time
            pch = list(pair_cnts.values())
            heapify_max(pch)
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
            # Update mapping pair -> count, pair -> pretokens and pair count
            # max heap accordingly
            _update_iteration_states(
                merged_p, pair_to_pretokens, pretoken_info, pair_cnts, pch
            )
            # The (old) merged pair is no longer needed so drop its entry
            pair_to_pretokens.pop(merged_p)
            if not pch:
                log.warning(
                    f"Cannot merge further as token pairs have run out. Actual vocab size: {len(vocab)} Expected: {vocab_size}"
                )
                break
        # print(f'DEBUG: heap before removing the merged pair: {pch}')
        merged_pc = heappop_max(pch)
        merged_p = merged_pc.pair
        merged_token = b"".join(merged_p)
        new_token_id = len(vocab)
        vocab[new_token_id] = merged_token
        merges.append(merged_p)
        log.debug(
            f"Merging pair {merged_p} of count {pair_cnts[merged_p].cnt} to new token {new_token_id}"
        )
        # print(f'DEBUG: pair_cnts = {pair_cnts}\npair_to_pretokens = {debug_pair_to_pretoken_info}')
        # The merged pair is no longer needed in pari_cnts so drop its entry
        pair_cnts.pop(merged_p)

    return vocab, merges


def _update_iteration_states(
    p: tuple[bytes, bytes],
    pair_to_pretokens: DefaultDict[tuple[bytes, bytes], set[str]],
    pretoken_info: dict[str, Pretoken],
    pair_cnts: dict[tuple[bytes, bytes], PairCount],
    pch: list[PairCount],
):
    """
    p: merged pair

    Spec:
    Invalidate PairCount values in pch whose pair's count has been updated,
    besides updating other states. After a pair's count has been fully updated,
    create a new PairCount value and push it to heap. Same for new pairs
    created from merging the merged token.
    """
    new_pcs: dict[tuple[bytes, bytes], PairCount] = {}
    for pt in pair_to_pretokens[p]:
        ptv = pretoken_info[pt]
        # replace all non-overlapping occurrences of token pair w/ new token
        old = ptv.seq
        idx, ln = 0, len(old)
        new = []
        merged_token = b"".join(p)
        # print(f'DEBUG: updating token sequence - merged pair {pair} pretoken str: "{pt}" token seq: {old}')
        while idx < ln:
            if idx < ln - 1 and old[idx] == p[0] and old[idx + 1] == p[1]:
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
                        # NOTE default value is returned only when we see this
                        # pair in the loop for the 1st time.
                        pc = new_pcs.get(p_gone, pair_cnts[p_gone].copy())
                        pc.cnt -= ptv.cnt
                        new_pcs[p_gone] = pc
                        # Invalidate the PairCount value from pair_cnts as its
                        # count has changed. This will later help us discard
                        # invalid PairCount values in the heap. NOTE this will
                        # NOT break the loose ordering b/w items in heap; Updating
                        # the count value of items in heap will, so avoid it.
                        pair_cnts[p_gone].valid = False
                        if pc.cnt == 0:
                            # p_gone's gone for real; Remove its states to free
                            # up mem space for future use
                            pair_cnts.pop(p_gone)
                            new_pcs.pop(p_gone)
                            pair_to_pretokens.pop(p_gone)

                    new_p_w_merge_token = (old[idx - 1], merged_token)
                    new_pc_w_merge_token = new_pcs.get(
                        new_p_w_merge_token, PairCount(new_p_w_merge_token, 0)
                    )
                    new_pc_w_merge_token.cnt += ptv.cnt
                    new_pcs[new_p_w_merge_token] = new_pc_w_merge_token
                    pair_to_pretokens[new_p_w_merge_token].add(pt)
                    # print(f'DEBUG: new pair w/ merged token {new_p_w_merge_token} - merged token {merged_token}')

                if idx + 2 < ln:
                    p_gone = (old[idx + 1], old[idx + 2])
                    # similar to paradigm above
                    if p_gone in pair_cnts:
                        pc = new_pcs.get(p_gone, pair_cnts[p_gone].copy())
                        pc.cnt -= ptv.cnt
                        new_pcs[p_gone] = pc
                        pair_cnts[p_gone].valid = False
                        if pair_cnts[p_gone] == 0:
                            pair_cnts.pop(p_gone)
                            new_pcs.pop(p_gone)
                            pair_to_pretokens.pop(p_gone)

                    new_p_w_merge_token = (merged_token, old[idx + 2])
                    new_pc_w_merge_token = new_pcs.get(
                        new_p_w_merge_token, PairCount(new_p_w_merge_token, 0)
                    )
                    new_pc_w_merge_token.cnt += ptv.cnt
                    new_pcs[new_p_w_merge_token] = new_pc_w_merge_token
                    pair_to_pretokens[new_p_w_merge_token].add(pt)
                    # print(f'DEBUG: new pair w/ merged token {new_p_w_merge_token} - merged token {merged_token}')
                idx += 2
            else:
                new.append(old[idx])
                idx += 1

        ptv.seq = new
    # Now we have collected pair -> count entries for:
    # 1. Existent pairs whose count has been decremented (not down to 0)
    # 2. New pairs built from merging the merged token
    # Now restore the (count and lexical) ordering among pairs by putting
    # these entries to pair_cnts and pair count max heap.
    for pair, pc in new_pcs.items():
        pair_cnts[pair] = pc
        heappush_max(pch, pc)
    # Finally pop the heap until we see a valid PairCount at heap top, which is
    # the next merged pair
    while pch and (not pch[0].valid or pch[0].cnt == 0):
        heappop_max(pch)


class PairCount:
    """
    Serves as a container of pair's info needed in BPE iteration run. Most
    importantly, it acts as 1/ sort key to find token pair of highest count
    and largest lexical order and 2/ a bag to carry important metadata, e.g.
    the `valid` flag which initiates to True and flipped to False if the pair's
    count changed during BPE iteration.

    NOTE this is a useful way to encapsulate complex comparison logic which
    cannot fit into a one-liner lambda function:

    Suppose pairs is a list of token pairs.
    Before:
    sorted(pairs, key=lambda p: # cannot fit in logic to first compare count then lexical order! ...)
    After:
    sorted(pairs, key=lambda p: PairCount(pair=p, cnt=pair_cnts[pair]))
    """

    def __init__(self, pair: tuple[bytes, bytes], cnt: int) -> None:
        self.pair = pair
        self.cnt = cnt
        # Check this when we pop a PairCount value from max heap
        # Discard the value if it is false. NOTE value of this field
        # shall NOT influence a PairCount's ordering in heap -- We
        # shall refrain from deliberately violating heap's invariance.
        self.valid = True

    def __lt__(self, other: "PairCount") -> bool:
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
        return str((self.pair, self.valid, self.cnt))

    def copy(self) -> PairCount:
        "Return a copy of itself."
        return PairCount(self.pair, self.cnt)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding=UTF8) as f:
        pretokens: dict[str, int] = json.load(f)

    return bpe_optimized(pretokens, vocab_size, special_tokens)


# @click.command(name="bpe")
# @click.argument("infile", type=click.Path(exists=True, path_type=Path))
# @click.argument("size", type=click.INT)
# def run(infile: Path, size: int):
#    """
#    INFILE the file path of mapping of pretokens to their counts in text
#    corpus.
#
#    SIZE the size of vocabulary that byte-pair encoding logic will build.
#    """
#    train_bpe(
#        str(infile),
#        vocab_size=size,
#        special_tokens=["<|endoftext|>"],
#    )
#
#
# if __name__ == "__main__":
#    run()
