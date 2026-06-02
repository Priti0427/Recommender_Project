# Popularity baseline: score each candidate by training-set interaction count.
#
# Stronger than Random but ignores the user -- recommends the same top-K
# regardless of who's asking. A useful upper bound on what user-agnostic
# popularity bias alone can achieve on this slate. If CaFE-Rec barely beats
# Popularity, the model has not learned anything user-specific.
#
# Usage:
#   python -m baselines.run_popularity --dataset amazon_movies \
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
    ap = argparse.ArgumentParser(description="Popularity baseline.")
    add_common_args(ap)
    args = ap.parse_args()

    cfg = resolve_config(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df, val_df, test_df, _ = load_splits(cfg.processed_dir)
    test_df = maybe_subsample(test_df, args.sample_test, seed=args.seed)
    _, num_items = num_users_items(train_df, val_df, test_df)

    # Item popularity from train only -- test/val leakage would inflate metrics.
    counts = np.zeros(num_items, dtype=np.float64)
    np.add.at(counts, train_df["item_idx"].to_numpy(dtype=np.int64), 1.0)
    # Log-scale so a long tail doesn't dominate scoring; preserves ranking
    # but produces more visually-meaningful per-example score differences.
    pop = torch.from_numpy(np.log1p(counts)).float()

    eval_ds = build_eval_dataset(
        test_df, train_df, num_items=num_items,
        num_negatives=cfg.num_neg_samples, seed=args.seed, val_df=val_df,
    )

    def scorer(u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return pop[i.cpu().long()]

    aggregate, per_example = rank_and_score(
        scorer, eval_ds, cfg, batch_size=args.batch_size,
        device="cpu", desc="popularity",
    )

    print("\n=== Popularity baseline ===")
    for k, v in aggregate.items():
        print(f"  {k:<12} = {v:.4f}")

    save_results(
        args.out, name="popularity",
        dataset_name=cfg.dataset_name,
        aggregate=aggregate,
        per_example=per_example,
        extras={"seed": args.seed, "num_negatives": cfg.num_neg_samples,
                "n_test_rows": len(test_df),
                "n_items_with_zero_train_interactions":
                    int((counts == 0).sum())},
    )


if __name__ == "__main__":
    main()
