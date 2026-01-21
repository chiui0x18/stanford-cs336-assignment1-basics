import json
from collections import Counter, defaultdict
import logging
from collections.abc import Iterable, Iterator

# from heapq import heapify_max, heappush_max, heappop_max  # requires python3.14
from heapq import heapify, heappush, heappop  # before python3.14
import math
from itertools import chain
from cs336_basics.log import get_logger  # logging logic in local module
from cs336_basics.pretokenizer import PRE_TOKENIZE_PAT
import regex as re

# Comment out imports like such which not needed by UTs to speed things up
# TODO move the CLI logic to a different py module instead?
# from pathlib import Path
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
        self.pair_to_pretokens: defaultdict[
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
        # Below works for python3.14 where we use heapq's max heap logic
        # if self.cnt != other.cnt:
        #    return self.cnt < other.cnt
        ## A tie on count; Break it by lexical ordering
        # return self.pair < other.pair

        # Counterintuitive it seems, it is to work around the fact that
        # python version < 3.14 doesn't have built in max heap logic and we
        # need this to use a min heap as if it is a max heap.
        if self.cnt != other.cnt:
            return self.cnt > other.cnt
        # A tie on count; Break it by lexical ordering
        return self.pair > other.pair

    def __repr__(self) -> str:
        return str((self.pair, self.valid, self.cnt))

    def copy(self) -> "PairCount":
        "Return a copy of itself."
        return PairCount(self.pair, self.cnt)


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
    pair_to_pretokens: defaultdict[tuple[bytes, bytes], set[str]] = defaultdict(set)
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
            # heapify_max(pch) # requires python3.14
            heapify(pch)
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
        # merged_pc = heappop_max(pch) # requires python3.14
        merged_pc = heappop(pch)
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
    pair_to_pretokens: defaultdict[tuple[bytes, bytes], set[str]],
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
        # heappush_max(pch, pc) # requires python3.14
        heappush(pch, pc)
    # Finally pop the heap until we see a valid PairCount at heap top, which is
    # the next merged pair
    while pch and (not pch[0].valid or pch[0].cnt == 0):
        # heappop_max(pch) # requires python3.14
        heappop(pch)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, encoding=UTF8) as f:
        pretokens: dict[str, int] = json.load(f)

    return bpe_optimized(pretokens, vocab_size, special_tokens)


class Tokenizer:
    """
    Tokenizer encodes given text to token sequence and decodes given token
    sequence to text.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Spec:
        Keep vocab for decoding token sequence to text.
        Keep merges for encoding text to token sequence.
        Need a way to look up a token's index given its bytes representation,
        so need to build a reverse bytes -> int mapping from vocab.
        For special tokens, first filter out those already existed in the
        reversed index built from vocab, then assign new id to the remaining
        ones and add mapping to both vocab and the reversed index, starting
        from |vocab|.
        """
        self.vocab = vocab
        # Merges need to proceed in the same order as merged pair creation so
        # use index as ordering indicator -- A pass to find token pair to merge
        # in a pretoken will need to find the pair of smallest index in mapping
        # below.
        self.merges: dict[tuple[bytes, bytes], int] = {
            p: idx for idx, p in enumerate(merges)
        }
        # the reverse bytes -> int mapping from given vocab
        self.bytes_to_token: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.pretoken_to_tokens: dict[str, list[int]] = {}
        self.pretokenize_pat = PRE_TOKENIZE_PAT
        self.special_tokens_pat = None
        special_tokens_regexes = []
        if isinstance(special_tokens, list):
            # capture longer special tokens first so that we can treat
            # repeated occurrence of special token as a single token
            for t in sorted(special_tokens, reverse=True):
                # to capture each special token in pretokenization
                special_tokens_regexes.append(re.escape(t))
                tb = t.encode(UTF8)
                if tb not in self.bytes_to_token:
                    new_token_id = len(vocab)
                    vocab[new_token_id] = tb
                    self.bytes_to_token[tb] = new_token_id
                self.pretoken_to_tokens[t] = [self.bytes_to_token[tb]]

            if special_tokens_regexes:
                self.special_tokens_pat = re.compile("|".join(special_tokens_regexes))

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        # TODO what does the content look like in vocab_filepath and merges_filepath?
        with open(vocab_filepath) as f_vocab:
            with open(merges_filepath) as f_merges:
                # FIXME clawning for now
                vocab = json.load(f_vocab)
                merges = json.load(f_merges)
                return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Spec:

        Pretokenize text. The result shall be:
        1. A mapping of pretokens -> list[int], whose value set to None
        2. A list of pretokens in order of given text.
        NOTE before pretokenization we must split text by given special
        tokens if any to avoid tokenizing the special tokens.

        For each pretoken in the mapping, find corresponding token list by
        merging the pretoken's byte-level presentation.

        Finally compose the return value by initing an empty list l, then for
        each pretoken in the list from point 2 above, extend l to
        include the corresponding token list.

        If encoding is called fairly frequently, it can be desirable for the
        Toekenizer instance to maintain a pretoken -> token list mapping as an
        attribute and use resulting cache effect to speed up encoding.
        """
        txt_and_spt_pairs_iter = [(text, "")]
        if self.special_tokens_pat:
            # Python has handy, comprehensive builtins for dealing w/ iteration:
            # https://docs.python.org/3/library/itertools.html
            special_tokens_iter = chain(
                map(
                    lambda m: text[m.start() : m.end()],
                    re.finditer(self.special_tokens_pat, text),
                ),
                # Below is necessary tomake the loop cover the last txt piece
                # after splitting the given text by special tokens
                [""],
            )
            txt_and_spt_pairs_iter = zip(
                re.splititer(self.special_tokens_pat, text),
                special_tokens_iter,
            )

        tokens = []
        for txt, spt in txt_and_spt_pairs_iter:
            # TODO what if txt is empty str?
            for m in re.finditer(self.pretokenize_pat, txt):
                pt = txt[m.start() : m.end()]
                if pt in self.pretoken_to_tokens:
                    tokens.extend(self.pretoken_to_tokens[pt])
                    continue
                # TODO pt not in pretoken -> token list mapping; Compute token list
                # and cache. Start merging from byte-level tokens
                pt_tokens = [bytes([b]) for b in pt.encode(UTF8)]
                # Enumerate the pairs in ptb and replace, until no replacement can
                # be found. TODO finally cache the pretoken -> token list entry
                # to self.pretoken_to_tokens
                # Edge case: pretoken contains only 1 byte
                while True:
                    merged_p, merged_p_order = None, math.inf
                    for p in zip(pt_tokens, pt_tokens[1:]):
                        p_order = self.merges.get(p, math.inf)
                        if p_order is not math.inf and p_order < merged_p_order:
                            merged_p = p
                            merged_p_order = p_order
                    # Found pair to merge or it is still None. If it is latter
                    # break, as no new merged pair is found; Otherwise replace token list
                    # w/ one w/ the merged token.
                    if merged_p is None:
                        break
                    merged_token = b"".join(merged_p)
                    pt_tokens_w_merged_p = []
                    idx, ln = 0, len(pt_tokens)
                    while idx < ln:
                        if (
                            idx < ln - 1
                            and pt_tokens[idx] == merged_p[0]
                            and pt_tokens[idx + 1] == merged_p[1]
                        ):
                            pt_tokens_w_merged_p.append(merged_token)
                            idx += 2
                        else:
                            pt_tokens_w_merged_p.append(pt_tokens[idx])
                            idx += 1
                    # prepare for next merge run
                    pt_tokens = pt_tokens_w_merged_p
                # Now look up the id of final tokens given their bytes representation
                pt_tokens = [self.bytes_to_token[b] for b in pt_tokens]
                self.pretoken_to_tokens[pt] = pt_tokens
                # finally extend the merged tokens to final token list
                tokens.extend(pt_tokens)
            # append id of special token to token list
            if spt != "":
                tokens.extend(self.pretoken_to_tokens[spt])

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        More on generator see:
        - https://stackoverflow.com/a/1756156
        - https://wiki.python.org/moin/Generators
        Spec:

        Iterate the given iterable - For each txt in iterable:
            tokens = self.encode(txt)
            for t in tokens:
                yield t
        """
        for txt in iterable:
            yield from self.encode(txt)

    def decode(self, ids: list[int]) -> str:
        """
        Spec:
        Start w/ an empty byte array.
        For each token id in ids:
            Look up vocab to get the bytes for token identified by id
            Put the bytes into byte array
        Decode byte array directly to str (feasible?)
        """
        b = bytearray()
        for t_id in ids:
            b.extend(self.vocab[t_id])
        return b.decode(encoding=UTF8, errors="replace")


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
