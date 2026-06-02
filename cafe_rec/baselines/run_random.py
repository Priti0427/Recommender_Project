# Random baseline: scores each candidate uniformly at random.
#
# Provides the per-example lower bound on Recall@K / NDCG@K. Expected
# Recall@10 with 100 negatives + 1 positive is ~10/101 ~ 0.099. A degenerate
# model that returns this or worse on real data should be debugged before
# being scored against PETER / PEPLER / XRec.
#
# Usage:
#   python -m baselines.run_random --dataset amazon_movies \
#       --processed_dir cafe_rec/data/processed --out results/baselines
from __future__ import annotations

import argparse

import numpy as np
import torch

from baselines.common import (
    add_common_args,
    build_eval_dataset,
    load_splits,
    maybe_subsample,
    num_users_items,
    rank_and_score,
    resolve_config,
    save_results,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Random-scoring baseline.")
    add_common_args(ap)
    args = ap.parse_args()

    cfg = resolve_config(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df, val_df, test_df, _ = load_splits(cfg.processed_dir)
    test_df = maybe_subsample(test_df, args.sample_test, seed=args.seed)
    _, num_items = num_users_items(train_df, val_df, test_df)
    eval_ds = build_eval_dataset(
        test_df, train_df, num_items=num_items,
        num_negatives=cfg.num_neg_samples, seed=args.seed, val_df=val_df,
    )

    # Per-call randomness: same (u, i) can score differently across calls,
    # which is fine for a baseline -- a random model has no fixed function.
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def scorer(u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return torch.rand(u.numel(), generator=gen)

    aggregate, per_example = rank_and_score(
        scorer, eval_ds, cfg, batch_size=args.batch_size,
        device="cpu", desc="random",
    )

    print("\n=== Random baseline ===")
    for k, v in aggregate.items():
        print(f"  {k:<12} = {v:.4f}")

    save_results(
        args.out, name="random",
        dataset_name=cfg.dataset_name,
        aggregate=aggregate,
        per_example=per_example,
        extras={"seed": args.seed, "num_negatives": cfg.num_neg_samples,
                "n_test_rows": len(test_df)},
    )


if __name__ == "__main__":
    main()
