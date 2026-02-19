from typing import Any
from collections.abc import Iterator
from itertools import product
from pathlib import Path
from datetime import datetime
import torch
import numpy as np


from cs336_basics.train import train_loop
from cs336_basics.log import get_logger

log = get_logger("hpo")


def hyperparam_space(
    batch_sizes: Iterator[int],
    adamw_betas: Iterator[tuple[float, float]],  # each tuple is (beta1, beta2)
    weight_decay_schedule_cfgs: Iterator[dict],
    lr_schedule_cfgs: Iterator[dict],
    # TODO other hyperparams
) -> Iterator[dict[str, Any]]:
    """
    Spec:

    Each input param is a range of a given hyperparam type for optimization.

    Enumerate all possible combinations of hyperparams and yield them via generator.

    Enumeration order matters. Range of hyperparameters which we are interested in
        changing and seeing the resulting effect the most and soonest shall be placed
        in the innermost/rightmost of the list fed to `itertools.product`.
    """
    d_batch_sizes = [{"batch_size": b} for b in batch_sizes]
    d_adamw_betas = [{"betas": v} for v in adamw_betas]
    d_wd_schedule_cfgs = [
        {"wd_schedule_cfg": cfg} for cfg in weight_decay_schedule_cfgs
    ]
    d_lr_schedule_cfgs = [{"lr_schedule_cfg": cfg} for cfg in lr_schedule_cfgs]

    # filter out empty iterables so that product() can work
    iters = [
        it
        for it in [
            d_batch_sizes,
            d_adamw_betas,
            d_wd_schedule_cfgs,
            d_lr_schedule_cfgs,
        ]
        if it
    ]
    # merge kwargs from different dict in ds together and yield
    for ds in product(*iters):
        yield {k: v for d in ds for k, v in d.items()}


vocab_size = 10000
ctx_len = 256
layers = 4
d_model = 512
heads = 16
d_ff = 1344
theta = 10000
tmax = 5000
metric_interval = 50
eval_interval = 50
grad_clip_norm = None
rand_seed = 2306125782

device = "cuda"
dtype = torch.float32

train_data_fp = "./data/TinyStoriesV2-GPT4-train.txt.vocab10000.tokenized.uint32.dat"
valid_data_fp = "./data/TinyStoriesV2-GPT4-valid.txt.vocab10000.tokenized.uint32.dat"
train_set = np.memmap(train_data_fp, dtype=np.uint32, mode="r")
valid_set = np.memmap(valid_data_fp, dtype=np.uint32, mode="r")


hps = hyperparam_space(
    batch_sizes=[64],
    adamw_betas=[
        (0.9, 0.99),
    ],
    weight_decay_schedule_cfgs=[
        {
            "type": "cosine_annealing_flx",
            "wd_init": 0.45,
            "wd_cool": 0.1,
            "wd_cold": 1e-3,
            "t_warm": 1500,
            "t_cool": 2500,
            "t_cold": 5000,
        },
    ],
    lr_schedule_cfgs=[
        {
            "type": "cosine_annealing_flx",
            "lr_max": 4e-3,
            "lr_cool": 1e-3,
            "lr_cold": 2.5e-4,
            "t_warm": 20,
            "t_cool": 2500,
            "t_cold": 5000,
        },
    ],
)

for idx, hp in enumerate(hps):
    log.info("Starting run %d w/ hyperparameters %s", idx, hp)
    run_name = f"tst-refactored-1"
    log_dir = Path(f"./runs/{run_name}")

    start = datetime.now()
    train_loop(
        train_set=train_set,
        validation_set=valid_set,
        vocab_size=vocab_size,
        context_len=ctx_len,
        num_layers=layers,
        d_model=d_model,
        num_heads=heads,
        d_ff=d_ff,
        rope_theta=theta,
        t_max=tmax,
        lr=4e-3,
        tensorboard_log_dir=log_dir,
        grad_clip_max_norm=None,
        rand_seed=rand_seed,
        device=device,
        dtype=dtype,
        eval_interval=eval_interval,
        autograd_detect_anomaly=False,
        metric_interval=metric_interval,
        metric_grad_norm=False,
        **hp,
    )
    d_train = datetime.now() - start
    log.info("Training run took %s", d_train)
