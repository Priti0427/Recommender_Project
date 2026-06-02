# BPR-MF baseline: matrix-factorization trained with BPR. Ranking-only
# (no text head), so this is the natural CF lower bound for explainable
# methods and a sanity check that our ranking eval is wired correctly.
#
# Implementation notes:
#   * Same embedding_dim as CaFE-Rec (config.embedding_dim) for a fair
#     capacity comparison on the recommendation side.
#   * Per-user negative sampling, identical to InteractionDataset.
#   * Eval slate is built with the same (seed, num_negatives) as ours, so
#     paired Wilcoxon over per-example arrays is valid.
#
# Usage:
#   python -m baselines.run_bprmf --dataset amazon_movies \
#       --processed_dir cafe_rec/data/processed --out results/baselines \
#       --epochs 10 --batch_size 1024
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from data.dataset import InteractionDataset, interaction_collate_fn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BPRMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(model: BPRMF, train_df, num_items: int, *,
          epochs: int, batch_size: int, lr: float, weight_decay: float,
          device: str, seed: int) -> None:
    ds = InteractionDataset(train_df, num_items=num_items, seed=seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=interaction_collate_fn,
        num_workers=0, drop_last=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device).train()
    for ep in range(epochs):
        t0 = time.time()
        total = 0.0; n_batches = 0
        pbar = tqdm(loader, desc=f"bpr-mf ep {ep:02d}", leave=False)
        for users, pos, neg in pbar:
            users = users.to(device); pos = pos.to(device); neg = neg.to(device)
            s_pos = model.score(users, pos)
            s_neg = model.score(users, neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()); n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"[bpr-mf] epoch {ep:02d} loss={total / max(n_batches, 1):.4f} "
              f"in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BPR-MF baseline.")
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--train_batch_size", type=int, default=1024,
                    help="Batch size for BPR training (separate from eval batch).")
    ap.add_argument("--sample_train", type=int, default=0,
                    help="If > 0, subsample training rows (smoke test).")
    args = ap.parse_args()

    cfg = resolve_config(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = cfg.device

    train_df, val_df, test_df, _ = load_splits(cfg.processed_dir)
    train_df_eval = train_df  # full train set used for negatives-exclusion in eval
    train_df = maybe_subsample(train_df, args.sample_train, seed=args.seed)
    test_df = maybe_subsample(test_df, args.sample_test, seed=args.seed)
    num_users, num_items = num_users_items(train_df_eval, val_df, test_df)

    model = BPRMF(num_users, num_items, dim=cfg.embedding_dim)
    print(f"[bpr-mf] {sum(p.numel() for p in model.parameters()):,} params "
          f"({num_users:,} users, {num_items:,} items, dim={cfg.embedding_dim}) "
          f"on {device}")

    train(model, train_df, num_items=num_items,
          epochs=args.epochs, batch_size=args.train_batch_size, lr=args.lr,
          weight_decay=args.weight_decay, device=device, seed=args.seed)

    eval_ds = build_eval_dataset(
        test_df, train_df_eval, num_items=num_items,
        num_negatives=cfg.num_neg_samples, seed=args.seed, val_df=val_df,
    )
    model.eval()

    @torch.no_grad()
    def scorer(u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        u = u.to(device); i = i.to(device)
        return model.score(u, i).cpu()

    aggregate, per_example = rank_and_score(
        scorer, eval_ds, cfg, batch_size=args.batch_size,
        device=device, desc="bpr-mf eval",
    )

    print("\n=== BPR-MF baseline ===")
    for k, v in aggregate.items():
        print(f"  {k:<12} = {v:.4f}")

    save_results(
        args.out, name="bprmf",
        dataset_name=cfg.dataset_name,
        aggregate=aggregate,
        per_example=per_example,
        extras={
            "seed": args.seed, "epochs": args.epochs,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "train_batch_size": args.train_batch_size,
            "embedding_dim": cfg.embedding_dim,
            "num_negatives": cfg.num_neg_samples,
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
        },
    )


if __name__ == "__main__":
    main()
