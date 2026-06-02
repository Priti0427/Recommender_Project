# PEPLER baseline (Li, Zhang, Chen, Ren -- TOIS 2023).
#
# "Personalized Prompt Learning for Explainable Recommendation."
# Architecture: a pretrained LM (GPT-2 in the paper) augmented with
# continuous prompts derived from MF user/item embeddings; joint training of
# rating regression and explanation LM. The "soft-prompt" variant (PEPLER)
# is implemented here; PEPLER-D (discrete prompts) is not.
#
# Differences vs. upstream (lileipisces/PEPLER):
#   * Backbone: Flan-T5-small instead of GPT-2 small. T5 is what our
#     pipeline already pins, so this keeps the comparison apples-to-apples
#     with CaFE-Rec (same LM family, same tokenizer, same text metrics).
#   * Prompts are projected from a 64-dim MF embedding into 4 soft-prefix
#     tokens of T5 d_model dim, prepended to the encoder input embeddings.
#   * Rating head is a 2-layer MLP over [u_emb || i_emb]; identical to the
#     paper's recommendation module.
#
# Vendor-in path: set PEPLER_USE_UPSTREAM=1 and place the upstream repo at
# baselines/pepler_src/; _build_model() will use it instead.
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from baselines.common import (
    add_common_args,
    build_eval_dataset,
    compute_text_metrics,
    load_splits,
    maybe_subsample,
    num_users_items,
    rank_and_score,
    resolve_config,
    save_results,
)


PREFIX_LEN = 4   # number of soft tokens; matches PEPLER's default L


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PEPLERDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 64):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["review_text"]) if row["review_text"] else ""
        enc = self.tok(text, max_length=self.max_length, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "user": torch.tensor(int(row["user_idx"]), dtype=torch.long),
            "item": torch.tensor(int(row["item_idx"]), dtype=torch.long),
            "rating": torch.tensor(float(row.get("rating", 0.0)), dtype=torch.float),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PEPLER(nn.Module):
    def __init__(self, num_users: int, num_items: int,
                 lm_name: str = "google/flan-t5-small",
                 mf_dim: int = 64, prefix_len: int = PREFIX_LEN):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, mf_dim)
        self.item_emb = nn.Embedding(num_items, mf_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        self.d_model = self.lm.config.d_model
        self.prefix_len = prefix_len
        # Project [u || i] into prefix_len * d_model.
        self.prefix_proj = nn.Sequential(
            nn.Linear(2 * mf_dim, prefix_len * self.d_model // 2),
            nn.Tanh(),
            nn.Linear(prefix_len * self.d_model // 2, prefix_len * self.d_model),
        )
        # Rating head over MF embeddings (no LM signal).
        self.rating_head = nn.Sequential(
            nn.Linear(2 * mf_dim, mf_dim), nn.ReLU(), nn.Linear(mf_dim, 1),
        )

    def soft_prefix(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user); i = self.item_emb(item)
        pre = self.prefix_proj(torch.cat([u, i], dim=-1))
        return pre.view(user.size(0), self.prefix_len, self.d_model)

    def rate(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user); i = self.item_emb(item)
        return self.rating_head(torch.cat([u, i], dim=-1)).squeeze(-1)

    def forward(self, user, item, input_ids, attention_mask):
        """Train the decoder to GENERATE the review from the soft prefix.

        The encoder sees ONLY the prefix; the review is the decoder target
        (labels). Do NOT put the review words in the encoder input -- that
        turns the model into an autoencoder that copies the review out of its
        own input, so at generation time (prefix only, no review) it emits EOS
        immediately and produces empty text (BLEU=0)."""
        prefix = self.soft_prefix(user, item)
        prefix_mask = torch.ones(prefix.size(0), prefix.size(1),
                                 device=prefix.device, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.lm.config.pad_token_id] = -100
        out = self.lm(inputs_embeds=prefix, attention_mask=prefix_mask,
                      labels=labels)
        rating = self.rate(user, item)
        return out.loss, rating

    @torch.no_grad()
    def generate(self, user, item, max_length: int = 32) -> torch.Tensor:
        prefix = self.soft_prefix(user, item)
        prefix_mask = torch.ones(prefix.size(0), prefix.size(1),
                                 device=prefix.device, dtype=torch.long)
        return self.lm.generate(
            inputs_embeds=prefix, attention_mask=prefix_mask,
            max_length=max_length, num_beams=1, do_sample=False,
        )


def _build_model(num_users: int, num_items: int, lm_name: str) -> nn.Module:
    if os.environ.get("PEPLER_USE_UPSTREAM"):
        try:
            from baselines.pepler_src import PEPLER as Upstream  # type: ignore
            return Upstream(num_users, num_items, lm_name=lm_name)
        except ImportError:
            print("[pepler] PEPLER_USE_UPSTREAM set but baselines/pepler_src/ "
                  "not importable; falling back to in-file impl.")
    return PEPLER(num_users, num_items, lm_name=lm_name)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(model: PEPLER, train_df: pd.DataFrame, tokenizer, *,
          epochs: int, batch_size: int, lr: float, device: str,
          lambda_r: float = 0.1, max_length: int = 64) -> None:
    ds = PEPLERDataset(train_df, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device).train()
    for ep in range(epochs):
        t0 = time.time()
        sum_lm = sum_r = 0.0; n = 0
        pbar = tqdm(loader, desc=f"pepler ep {ep:02d}", leave=False)
        for batch in pbar:
            u = batch["user"].to(device); i = batch["item"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            r = batch["rating"].to(device)
            lm_loss, rating = model(u, i, ids, mask)
            reg = F.mse_loss(rating, r)
            loss = lm_loss + lambda_r * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sum_lm += float(lm_loss.item()); sum_r += float(reg.item()); n += 1
            pbar.set_postfix(lm=f"{lm_loss.item():.3f}", r=f"{reg.item():.3f}")
        print(f"[pepler] ep {ep:02d}  lm={sum_lm/max(n,1):.4f}  "
              f"reg={sum_r/max(n,1):.4f}  in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_texts(model: PEPLER, df: pd.DataFrame, tokenizer, device: str,
                   sample_limit: int = 500, max_length: int = 32
                   ) -> Tuple[List[str], List[str]]:
    if sample_limit and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=0).reset_index(drop=True)
    preds: List[str] = []; refs: List[str] = []
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="pepler gen"):
        u = torch.tensor([int(row["user_idx"])], device=device)
        i = torch.tensor([int(row["item_idx"])], device=device)
        out = model.generate(u, i, max_length=max_length)
        preds.append(tokenizer.decode(out[0].cpu().tolist(),
                                      skip_special_tokens=True))
        refs.append(str(row.get("review_text", "")))
    return preds, refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PEPLER baseline.")
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train_batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--lambda_r", type=float, default=0.1)
    ap.add_argument("--lm_name", default="google/flan-t5-small",
                    help="HF model id for the LM backbone.")
    ap.add_argument("--text_sample_limit", type=int, default=500)
    ap.add_argument("--sample_train", type=int, default=0)
    args = ap.parse_args()

    cfg = resolve_config(args)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = cfg.device

    train_df, val_df, test_df, _ = load_splits(cfg.processed_dir)
    train_df_eval = train_df
    train_df = maybe_subsample(train_df, args.sample_train, seed=args.seed)
    test_df = maybe_subsample(test_df, args.sample_test, seed=args.seed)
    num_users, num_items = num_users_items(train_df_eval, val_df, test_df)

    tokenizer = T5Tokenizer.from_pretrained(args.lm_name)
    model = _build_model(num_users, num_items, lm_name=args.lm_name)
    print(f"[pepler] {sum(p.numel() for p in model.parameters()):,} params on {device}")

    train(model, train_df, tokenizer,
          epochs=args.epochs, batch_size=args.train_batch_size, lr=args.lr,
          device=device, lambda_r=args.lambda_r, max_length=args.max_length)

    eval_ds = build_eval_dataset(
        test_df, train_df_eval, num_items=num_items,
        num_negatives=cfg.num_neg_samples, seed=args.seed, val_df=val_df,
    )
    model.eval()

    @torch.no_grad()
    def scorer(u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return model.rate(u.to(device), i.to(device)).cpu()

    aggregate, per_example = rank_and_score(
        scorer, eval_ds, cfg, batch_size=args.batch_size,
        device=device, desc="pepler eval",
    )

    preds, refs = generate_texts(
        model, test_df, tokenizer, device,
        sample_limit=args.text_sample_limit, max_length=32,
    )
    aggregate.update(compute_text_metrics(preds, refs, device=device))
    aggregate.setdefault("sufficiency", None)
    aggregate.setdefault("comprehensiveness", None)
    aggregate.setdefault("CF@10", None)

    print("\n=== PEPLER baseline ===")
    for k, v in aggregate.items():
        print(f"  {k:<14} = {v if v is None else f'{v:.4f}'}")

    save_results(
        args.out, name="pepler",
        dataset_name=cfg.dataset_name,
        aggregate=aggregate,
        per_example=per_example,
        extras={
            "epochs": args.epochs, "lr": args.lr,
            "train_batch_size": args.train_batch_size,
            "max_length": args.max_length, "lambda_r": args.lambda_r,
            "lm_name": args.lm_name, "prefix_len": PREFIX_LEN,
            "n_train_rows": len(train_df), "n_test_rows": len(test_df),
        },
    )


if __name__ == "__main__":
    main()
