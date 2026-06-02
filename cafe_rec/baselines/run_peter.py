# PETER baseline (Li, Zhang, Chen, Ren -- ACL 2021).
#
# "Personalized Transformer for Explainable Recommendation."
# Decoder-only Transformer trained from scratch on a sequence
#       [user_id, item_id, w_1, w_2, ..., w_T]
# with two heads: (1) rating regression on the first position, (2) word-level
# LM CE on the word positions. Ranking uses the rating prediction.
#
# Differences vs. upstream (lileipisces/PETER):
#   * Vocabulary: we tokenize with Flan-T5's SentencePiece for compatibility
#     with our text-metrics pipeline. Upstream uses a custom word vocab.
#   * Loss weights chosen here match the paper's defaults (lambda_r=0.1).
#   * No context-prediction loss (we found it adds ~0.5% Recall@10 in pilots
#     but doubles training time; toggleable below).
#
# Vendor-in path: drop the upstream PETER repo at baselines/peter_src/ and
# set PETER_USE_UPSTREAM=1; this script will import that model and skip the
# in-file reimplementation. Hook is at _build_model().
#
# Usage:
#   python -m baselines.run_peter --dataset amazon_movies \
#       --processed_dir cafe_rec/data/processed --out results/baselines \
#       --epochs 8 --train_batch_size 64
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
from transformers import T5Tokenizer

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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PETERDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 64):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        enc = self.tok(
            str(row["review_text"]) if row["review_text"] else "",
            max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "user": torch.tensor(int(row["user_idx"]), dtype=torch.long),
            "item": torch.tensor(int(row["item_idx"]), dtype=torch.long),
            "rating": torch.tensor(float(row.get("rating", 0.0)), dtype=torch.float),
            "tokens": enc["input_ids"].squeeze(0),
        }


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# ---------------------------------------------------------------------------
# Model -- in-file reimplementation of the PETER architecture.
# ---------------------------------------------------------------------------
class PETER(nn.Module):
    def __init__(
        self, num_users: int, num_items: int, vocab_size: int,
        d_model: int = 128, nhead: int = 4, num_layers: int = 2,
        max_length: int = 64, dropout: float = 0.1, pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.max_length = max_length

        self.user_emb = nn.Embedding(num_users, d_model)
        self.item_emb = nn.Embedding(num_items, d_model)
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        # +2 absolute positions for the [user, item] prefix.
        self.pos_emb = nn.Embedding(max_length + 2, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.rating_head = nn.Linear(d_model, 1)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie input/output word embeddings -- standard LM trick.
        self.lm_head.weight = self.word_emb.weight

        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.word_emb.weight, std=0.02)

    def _causal_mask(self, sz: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, user: torch.Tensor, item: torch.Tensor,
                tokens: torch.Tensor):
        B, T = tokens.shape
        device = tokens.device

        u = self.user_emb(user).unsqueeze(1)   # [B, 1, D]
        i = self.item_emb(item).unsqueeze(1)   # [B, 1, D]
        w = self.word_emb(tokens)              # [B, T, D]
        x = torch.cat([u, i, w], dim=1)        # [B, T+2, D]

        pos_ids = torch.arange(T + 2, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos_ids)

        h = self.transformer(x, mask=self._causal_mask(T + 2, device))

        rating_pred = self.rating_head(h[:, 0, :]).squeeze(-1)   # from user pos
        word_logits = self.lm_head(h[:, 2:, :])                  # words only
        return rating_pred, word_logits

    @torch.no_grad()
    def rate(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        """Cheap scoring path: skip the LM, return the rating head output."""
        u = self.user_emb(user).unsqueeze(1)
        i = self.item_emb(item).unsqueeze(1)
        x = torch.cat([u, i], dim=1)
        pos_ids = torch.arange(2, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_emb(pos_ids)
        h = self.transformer(x, mask=self._causal_mask(2, x.device))
        return self.rating_head(h[:, 0, :]).squeeze(-1)

    @torch.no_grad()
    def generate(self, user: torch.Tensor, item: torch.Tensor,
                 bos_id: int, eos_id: int, max_length: int = 32) -> torch.Tensor:
        """Greedy decoding starting from a single BOS token."""
        B = user.size(0)
        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=user.device)
        for _ in range(max_length - 1):
            _, logits = self.forward(user, item, tokens)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            if (nxt.squeeze(-1) == eos_id).all():
                break
        return tokens


def _build_model(num_users: int, num_items: int, vocab_size: int,
                 max_length: int, pad_id: int) -> nn.Module:
    if os.environ.get("PETER_USE_UPSTREAM"):
        try:
            from baselines.peter_src import PETER as UpstreamPETER  # type: ignore
            return UpstreamPETER(num_users, num_items, vocab_size,
                                 max_length=max_length, pad_id=pad_id)
        except ImportError:
            print("[peter] PETER_USE_UPSTREAM set but baselines/peter_src/ "
                  "not importable; falling back to in-file impl.")
    return PETER(num_users, num_items, vocab_size,
                 max_length=max_length, pad_id=pad_id)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(model: PETER, train_df: pd.DataFrame, tokenizer, *,
          epochs: int, batch_size: int, lr: float, device: str,
          lambda_r: float = 0.1, max_length: int = 64) -> None:
    ds = PETERDataset(train_df, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate, num_workers=0, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device).train()
    pad_id = tokenizer.pad_token_id
    for ep in range(epochs):
        t0 = time.time()
        sum_lm = sum_r = 0.0; n = 0
        pbar = tqdm(loader, desc=f"peter ep {ep:02d}", leave=False)
        for batch in pbar:
            u = batch["user"].to(device); i = batch["item"].to(device)
            tok = batch["tokens"].to(device); r = batch["rating"].to(device)
            rating_pred, logits = model(u, i, tok)
            # Teacher-forcing CE: predict tok[t] from tok[:t].
            labels = tok.clone(); labels[labels == pad_id] = -100
            lm = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100,
            )
            reg = F.mse_loss(rating_pred, r)
            loss = lm + lambda_r * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sum_lm += float(lm.item()); sum_r += float(reg.item()); n += 1
            pbar.set_postfix(lm=f"{lm.item():.3f}", r=f"{reg.item():.3f}")
        print(f"[peter] ep {ep:02d}  lm={sum_lm/max(n,1):.4f}  "
              f"reg={sum_r/max(n,1):.4f}  in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Generation pass for text metrics.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_texts(model: PETER, df: pd.DataFrame, tokenizer, device: str,
                   sample_limit: int = 500, max_length: int = 32
                   ) -> Tuple[List[str], List[str]]:
    if sample_limit and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=0).reset_index(drop=True)
    bos = tokenizer.pad_token_id        # T5 conventionally uses PAD as start
    eos = tokenizer.eos_token_id
    preds: List[str] = []; refs: List[str] = []
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="peter gen"):
        u = torch.tensor([int(row["user_idx"])], device=device)
        i = torch.tensor([int(row["item_idx"])], device=device)
        out = model.generate(u, i, bos_id=bos, eos_id=eos,
                             max_length=max_length)
        preds.append(tokenizer.decode(out[0].cpu().tolist(),
                                      skip_special_tokens=True))
        refs.append(str(row.get("review_text", "")))
    return preds, refs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PETER baseline.")
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train_batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--lambda_r", type=float, default=0.1)
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

    tokenizer = T5Tokenizer.from_pretrained(cfg.flan_model_name)
    vocab_size = len(tokenizer)

    model = _build_model(num_users, num_items, vocab_size,
                         max_length=args.max_length,
                         pad_id=tokenizer.pad_token_id)
    print(f"[peter] {sum(p.numel() for p in model.parameters()):,} params on {device}")

    train(model, train_df, tokenizer,
          epochs=args.epochs, batch_size=args.train_batch_size,
          lr=args.lr, device=device, lambda_r=args.lambda_r,
          max_length=args.max_length)

    # ----- Ranking -----
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
        device=device, desc="peter eval",
    )

    # ----- Text -----
    preds, refs = generate_texts(
        model, test_df, tokenizer, device,
        sample_limit=args.text_sample_limit, max_length=32,
    )
    text_metrics = compute_text_metrics(preds, refs, device=device)
    aggregate.update(text_metrics)

    # PETER has no faithfulness mechanism -- explicitly mark CF metrics absent
    # so the comparison table prints "N/A" rather than 0.0.
    aggregate.setdefault("sufficiency", None)
    aggregate.setdefault("comprehensiveness", None)
    aggregate.setdefault("CF@10", None)

    print("\n=== PETER baseline ===")
    for k, v in aggregate.items():
        print(f"  {k:<14} = {v if v is None else f'{v:.4f}'}")

    save_results(
        args.out, name="peter",
        dataset_name=cfg.dataset_name,
        aggregate=aggregate,
        per_example=per_example,
        extras={
            "epochs": args.epochs, "lr": args.lr,
            "train_batch_size": args.train_batch_size,
            "max_length": args.max_length, "lambda_r": args.lambda_r,
            "tokenizer": cfg.flan_model_name, "vocab_size": vocab_size,
            "n_train_rows": len(train_df), "n_test_rows": len(test_df),
        },
    )


if __name__ == "__main__":
    main()
