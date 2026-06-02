# XRec baseline (Ma, Xia, Lin, Ren, Huang -- 2024).
#
# "XRec: Large Language Models for Explainable Recommendation."
# Architecture: collaborative embeddings from a graph CF model fed into an
# LLM via a (mixture-of-experts) adapter; LLM generates the explanation.
# No information bottleneck, no faithfulness loss -- this is structurally
# the closest non-faithful competitor to CaFE-Rec, and effectively also
# acts as an ablation "CaFE-Rec - {AspectSelector, L_faith, L_sparse}".
#
# Differences vs. upstream (HKUDS/XRec):
#   * LLM backbone: Flan-T5-base instead of Llama-2-7B. Llama would be the
#     paper-faithful choice but is impractical for Colab; we hold the LM
#     family constant with CaFE-Rec so the comparison isolates the
#     bottleneck+L_faith contribution rather than confounding with LLM scale.
#   * Adapter: 2-layer MLP, no mixture-of-experts (paper's MoE adds <1%
#     BLEU in their ablations on smaller backbones).
#   * Collaborative side: LightGCN (same K as CaFE-Rec, sharing the graph
#     propagation code from models.lightgcn) so embedding quality is matched.
#
# Vendor-in path: set XREC_USE_UPSTREAM=1 and place the upstream repo at
# baselines/xrec_src/.
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
from data.dataset import InteractionDataset, interaction_collate_fn
from models.lightgcn import LightGCN


PREFIX_LEN = 8   # XRec uses longer prefixes than PEPLER


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class XRecTextDataset(Dataset):
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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class XRec(nn.Module):
    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int, num_gcn_layers: int,
                 lm_name: str = "google/flan-t5-base",
                 prefix_len: int = PREFIX_LEN):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.lightgcn = LightGCN(
            num_users=num_users, num_items=num_items,
            embedding_dim=embedding_dim, num_layers=num_gcn_layers,
        )
        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        self.d_model = self.lm.config.d_model
        self.prefix_len = prefix_len
        self.adapter = nn.Sequential(
            nn.Linear(2 * embedding_dim, prefix_len * self.d_model // 2),
            nn.GELU(),
            nn.Linear(prefix_len * self.d_model // 2, prefix_len * self.d_model),
        )
        self.register_buffer(
            "_edge_index", torch.empty(2, 0, dtype=torch.long), persistent=False,
        )
        self._edge_index_ready = False

    def build_graph(self, train_df: pd.DataFrame) -> None:
        edge_index = LightGCN.build_edge_index(
            train_df, self.num_users, self.num_items,
        )
        device = next(self.parameters()).device
        self._edge_index = edge_index.to(device)
        self._edge_index_ready = True

    def get_embeddings(self):
        if not self._edge_index_ready:
            raise RuntimeError(
                "edge_index not built. Call model.build_graph(train_df) first."
            )
        return self.lightgcn(self._edge_index)

    def score(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        all_u, all_i = self.get_embeddings()
        return (all_u[user] * all_i[item]).sum(dim=-1)

    def soft_prefix(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        all_u, all_i = self.get_embeddings()
        u = all_u[user]; i = all_i[item]
        pre = self.adapter(torch.cat([u, i], dim=-1))
        return pre.view(user.size(0), self.prefix_len, self.d_model)

    def forward_text(self, user, item, input_ids, attention_mask):
        # Encoder sees ONLY the soft prefix; decoder is trained to GENERATE the
        # review (labels). Concatenating the review into the encoder input made
        # the model copy it (autoencoder) and produce empty text at generation
        # time -- see the same fix in run_pepler.PEPLER.forward.
        prefix = self.soft_prefix(user, item)
        prefix_mask = torch.ones(prefix.size(0), prefix.size(1),
                                 device=prefix.device, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.lm.config.pad_token_id] = -100
        return self.lm(inputs_embeds=prefix, attention_mask=prefix_mask,
                       labels=labels).loss

    @torch.no_grad()
    def generate(self, user, item, max_length: int = 32) -> torch.Tensor:
        prefix = self.soft_prefix(user, item)
        prefix_mask = torch.ones(prefix.size(0), prefix.size(1),
                                 device=prefix.device, dtype=torch.long)
        return self.lm.generate(
            inputs_embeds=prefix, attention_mask=prefix_mask,
            max_length=max_length, num_beams=1, do_sample=False,
        )


def _build_model(num_users: int, num_items: int, cfg, lm_name: str) -> nn.Module:
    if os.environ.get("XREC_USE_UPSTREAM"):
        try:
            from baselines.xrec_src import XRec as Upstream  # type: ignore
            return Upstream(num_users, num_items, lm_name=lm_name)
        except ImportError:
            print("[xrec] XREC_USE_UPSTREAM set but baselines/xrec_src/ "
                  "not importable; falling back to in-file impl.")
    return XRec(
        num_users, num_items,
        embedding_dim=cfg.embedding_dim, num_gcn_layers=cfg.num_gcn_layers,
        lm_name=lm_name,
    )


# ---------------------------------------------------------------------------
# Train (BPR pretrain -> joint text+BPR)
# ---------------------------------------------------------------------------
def train(model: XRec, train_df: pd.DataFrame, tokenizer, *,
          epochs: int, bpr_pretrain_epochs: int,
          text_batch_size: int, bpr_batch_size: int,
          lr: float, device: str, num_items: int,
          seed: int, max_length: int = 64) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    # ---------- (1) BPR pretraining on graph embeddings ----------
    bpr_ds = InteractionDataset(train_df, num_items=num_items, seed=seed)
    bpr_loader = DataLoader(
        bpr_ds, batch_size=bpr_batch_size, shuffle=True,
        collate_fn=interaction_collate_fn, num_workers=0, drop_last=True,
    )
    model.train()
    for ep in range(bpr_pretrain_epochs):
        t0 = time.time(); total = 0.0; n = 0
        pbar = tqdm(bpr_loader, desc=f"xrec bpr-pre {ep:02d}", leave=False)
        for users, pos, neg in pbar:
            users = users.to(device); pos = pos.to(device); neg = neg.to(device)
            s_pos = model.score(users, pos); s_neg = model.score(users, neg)
            loss = -F.logsigmoid(s_pos - s_neg).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            total += float(loss.item()); n += 1
            pbar.set_postfix(bpr=f"{loss.item():.4f}")
        print(f"[xrec] bpr-pretrain ep {ep:02d}  loss={total/max(n,1):.4f}  "
              f"in {time.time()-t0:.1f}s")

    # ---------- (2) Joint text + BPR ----------
    text_ds = XRecTextDataset(train_df, tokenizer, max_length=max_length)
    text_loader = DataLoader(
        text_ds, batch_size=text_batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0, drop_last=True,
    )
    for ep in range(epochs):
        t0 = time.time()
        sum_lm = sum_bpr = 0.0; n = 0
        bpr_iter = iter(bpr_loader)
        pbar = tqdm(text_loader, desc=f"xrec joint ep {ep:02d}", leave=False)
        for batch in pbar:
            u = batch["user"].to(device); i = batch["item"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lm_loss = model.forward_text(u, i, ids, mask)

            # Re-sample one BPR batch per text batch.
            try:
                bu, bp, bn = next(bpr_iter)
            except StopIteration:
                bpr_iter = iter(bpr_loader); bu, bp, bn = next(bpr_iter)
            bu = bu.to(device); bp = bp.to(device); bn = bn.to(device)
            s_pos = model.score(bu, bp); s_neg = model.score(bu, bn)
            bpr_loss = -F.logsigmoid(s_pos - s_neg).mean()

            loss = lm_loss + bpr_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sum_lm += float(lm_loss.item()); sum_bpr += float(bpr_loss.item()); n += 1
            pbar.set_postfix(lm=f"{lm_loss.item():.3f}", bpr=f"{bpr_loss.item():.3f}")
        print(f"[xrec] joint ep {ep:02d}  lm={sum_lm/max(n,1):.4f}  "
              f"bpr={sum_bpr/max(n,1):.4f}  in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_texts(model: XRec, df: pd.DataFrame, tokenizer, device: str,
                   sample_limit: int = 500, max_length: int = 32
                   ) -> Tuple[List[str], List[str]]:
    if sample_limit and len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=0).reset_index(drop=True)
    preds: List[str] = []; refs: List[str] = []
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="xrec gen"):
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
    ap = argparse.ArgumentParser(description="XRec baseline.")
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=3,
                    help="Joint text+BPR epochs.")
    ap.add_argument("--bpr_pretrain_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--text_batch_size", type=int, default=16)
    ap.add_argument("--bpr_batch_size", type=int, default=1024)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--lm_name", default="google/flan-t5-base")
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
    model = _build_model(num_users, num_items, cfg, lm_name=args.lm_name)
    model.to(device)                       # buffer/device sync before build_graph
    model.build_graph(train_df_eval)
    print(f"[xrec] {sum(p.numel() for p in model.parameters()):,} params on {device}")

    train(model, train_df, tokenizer,
          epochs=args.epochs, bpr_pretrain_epochs=args.bpr_pretrain_epochs,
          text_batch_size=args.text_batch_size,
          bpr_batch_size=args.bpr_batch_size, lr=args.lr,
          device=device, num_items=num_items, seed=args.seed,
          max_length=args.max_length)

    eval_ds = build_eval_dataset(
        test_df, train_df_eval, num_items=num_items,
        num_negatives=cfg.num_neg_samples, seed=args.seed, val_df=val_df,
    )
    model.eval()

    @torch.no_grad()
    def scorer(u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return model.score(u.to(device), i.to(device)).cpu()

    aggregate, per_example = rank_and_score(
        scorer, eval_ds, cfg, batch_size=args.batch_size,
        device=device, desc="xrec eval",
    )

    preds, refs = generate_texts(
        model, test_df, tokenizer, device,
        sample_limit=args.text_sample_limit, max_length=32,
    )
    aggregate.update(compute_text_metrics(preds, refs, device=device))
    aggregate.setdefault("sufficiency", None)
    aggregate.setdefault("comprehensiveness", None)
    aggregate.setdefault("CF@10", None)

    print("\n=== XRec baseline ===")
    for k, v in aggregate.items():
        print(f"  {k:<14} = {v if v is None else f'{v:.4f}'}")

    save_results(
        args.out, name="xrec",
        dataset_name=cfg.dataset_name,
        aggregate=aggregate,
        per_example=per_example,
        extras={
            "epochs": args.epochs,
            "bpr_pretrain_epochs": args.bpr_pretrain_epochs,
            "lr": args.lr,
            "text_batch_size": args.text_batch_size,
            "bpr_batch_size": args.bpr_batch_size,
            "max_length": args.max_length, "lm_name": args.lm_name,
            "prefix_len": PREFIX_LEN,
            "embedding_dim": cfg.embedding_dim,
            "num_gcn_layers": cfg.num_gcn_layers,
            "n_train_rows": len(train_df), "n_test_rows": len(test_df),
        },
    )


if __name__ == "__main__":
    main()
