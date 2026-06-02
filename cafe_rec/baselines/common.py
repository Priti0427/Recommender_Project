# Shared harness for baseline runners.
#
# Every baseline (random / popularity / BPR-MF / PETER / PEPLER / XRec) consumes
# the same processed splits as CaFE-Rec, runs against the same EvaluationDataset
# (so the candidate slate per row is bit-identical), and emits two artifacts in
# a directory chosen by --out:
#
#   <name>.json              aggregate metrics
#   <name>_per_example.npz   per-row arrays for paired Wilcoxon vs. ours
#
# evaluate.py.load_baseline_results() already understands this layout.
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make `from config import ...` resolve when run as `python -m baselines.run_*`
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from config import Config, CONFIG  # noqa: E402
from data.dataset import EvaluationDataset, evaluation_collate_fn  # noqa: E402
from evaluation.rec_metrics import ndcg_at_k, recall_at_k  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading -- mirror train.load_data() so baselines see identical splits.
# ---------------------------------------------------------------------------
def load_splits(processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    paths = {k: os.path.join(processed_dir, f"{k}.pkl") for k in ("train", "val", "test")}
    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run data.preprocess first.")
    vocab_path = os.path.join(processed_dir, "aspect_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"{vocab_path} not found. Run data.aspect_extract first.")

    def _load(p):
        with open(p, "rb") as f:
            return pickle.load(f)

    train_df = _load(paths["train"])
    val_df = _load(paths["val"])
    test_df = _load(paths["test"])
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return train_df, val_df, test_df, vocab


def num_users_items(*dfs: pd.DataFrame) -> Tuple[int, int]:
    nu = max(int(df["user_idx"].max()) for df in dfs) + 1
    ni = max(int(df["item_idx"].max()) for df in dfs) + 1
    return nu, ni


def build_eval_dataset(
    test_df: pd.DataFrame, train_df: pd.DataFrame, num_items: int,
    num_negatives: int = 100, seed: int = 42,
    val_df: Optional[pd.DataFrame] = None,
) -> EvaluationDataset:
    """Same slate every baseline -- (seed, num_negatives) must match what
    train.py uses to build the val/test EvaluationDataset."""
    return EvaluationDataset(
        test_df, train_df, num_items=num_items,
        num_negatives=num_negatives, seed=seed, val_df=val_df,
    )


# ---------------------------------------------------------------------------
# Ranking via a scorer callback.
# ---------------------------------------------------------------------------
# Scorer signature:
#     scorer(user_ids: LongTensor[B],
#            item_ids: LongTensor[B*C])           -> Tensor[B, C]
# where C = 1 + num_negatives. The harness handles all batching and metric
# aggregation; baselines only have to implement how to score (u, i) pairs.
Scorer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def rank_and_score(
    scorer: Scorer,
    eval_dataset: EvaluationDataset,
    config: Config,
    batch_size: int = 256,
    device: Optional[str] = None,
    desc: str = "rank",
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Run the eval slate through `scorer`, return (aggregate, per_example).

    Aggregate: {"Recall@K": float, "NDCG@K": float, ...} for each K in
               config.top_k_list.
    Per-example: {"Recall@K": np.ndarray[N], ...} (same keys, per row).
    """
    device = torch.device(device or config.device)
    loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=evaluation_collate_fn,
    )
    ks = list(config.top_k_list)
    agg = {f"Recall@{k}": [] for k in ks}
    agg.update({f"NDCG@{k}": [] for k in ks})

    for users, pos, negs in tqdm(loader, desc=desc):
        users = users.to(device); pos = pos.to(device); negs = negs.to(device)
        B = users.size(0)
        cands = torch.cat([pos.unsqueeze(1), negs], dim=1)   # [B, C]
        C = cands.size(1)
        u_rep = users.unsqueeze(1).expand(-1, C).reshape(-1)
        i_rep = cands.reshape(-1)
        scores = scorer(u_rep, i_rep).view(B, C).cpu().numpy()
        for row in scores:
            for k in ks:
                agg[f"Recall@{k}"].append(recall_at_k(row, 0, k))
                agg[f"NDCG@{k}"].append(ndcg_at_k(row, 0, k))

    per = {k: np.asarray(v, dtype=np.float32) for k, v in agg.items()}
    aggregate = {k: float(v.mean()) for k, v in per.items()}
    return aggregate, per


# ---------------------------------------------------------------------------
# Text metrics for explainable baselines (PETER / PEPLER / XRec).
# Reuse the same BLEU/ROUGE/BERTScore helpers CaFE-Rec uses, so the comparison
# is apples-to-apples.
# ---------------------------------------------------------------------------
def compute_text_metrics(
    preds: List[str], refs: List[str],
    device: str = "cpu",
    aspect_vocabulary: Optional[List[str]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        from evaluation.text_metrics import compute_bleu
        out.update({k: float(v) for k, v in compute_bleu(preds, refs).items()})
    except Exception as e:
        print(f"[text] BLEU skipped: {e}")
    try:
        from evaluation.text_metrics import compute_rouge
        out.update({k: float(v) for k, v in compute_rouge(preds, refs).items()})
    except Exception as e:
        print(f"[text] ROUGE skipped: {e}")
    try:
        from evaluation.text_metrics import compute_bertscore
        out.update({k: float(v) for k, v in compute_bertscore(preds, refs, device=device).items()})
    except Exception as e:
        print(f"[text] BERTScore skipped: {e}")
    if aspect_vocabulary:
        try:
            from evaluation.text_metrics import compute_fmr
            out.update({k: float(v) for k, v in compute_fmr(preds, aspect_vocabulary).items()})
        except Exception as e:
            print(f"[text] FMR skipped: {e}")
    return out


# ---------------------------------------------------------------------------
# Output writers.
# ---------------------------------------------------------------------------
def save_results(
    out_dir: str,
    name: str,
    dataset_name: str,
    aggregate: Dict[str, Any],
    per_example: Optional[Dict[str, np.ndarray]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """Write <name>.json and (optionally) <name>_per_example.npz.

    JSON shape (matches evaluate.py.load_baseline_results()):
        {<dataset_name>: {"Recall@10": ..., "NDCG@10": ..., ..., "_meta": {...}}}
    """
    os.makedirs(out_dir, exist_ok=True)
    payload = {dataset_name: {
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in aggregate.items()},
        "_meta": {
            "name": name, "dataset": dataset_name,
            "timestamp": int(time.time()), **(extras or {}),
        },
    }}
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] {json_path}")

    npz_path: Optional[str] = None
    if per_example:
        npz_path = os.path.join(out_dir, f"{name}_per_example.npz")
        np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in per_example.items()})
        print(f"[save] {npz_path}")
    return json_path, npz_path


# ---------------------------------------------------------------------------
# CLI helpers reused across baselines.
# ---------------------------------------------------------------------------
def add_common_args(parser) -> None:
    parser.add_argument("--dataset", default="amazon_movies")
    parser.add_argument("--processed_dir", default=None,
                        help="Override config.processed_dir.")
    parser.add_argument("--out", default="./results/baselines",
                        help="Directory to drop <name>.json and <name>_per_example.npz.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_test", type=int, default=0,
                        help="If > 0, sub-sample test rows (debugging only).")
    parser.add_argument("--seed", type=int, default=42)


def resolve_config(args) -> Config:
    cfg = CONFIG
    if getattr(args, "processed_dir", None):
        cfg = replace(cfg, processed_dir=args.processed_dir)
    cfg = replace(cfg, dataset_name=args.dataset)
    return cfg


def maybe_subsample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n and n > 0 and n < len(df):
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df
