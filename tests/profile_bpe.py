import time
from common import FIXTURES_PATH
from adapters import run_train_bpe
import cProfile


def profile_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()


if __name__ == "__main__":
    profile_bpe()
