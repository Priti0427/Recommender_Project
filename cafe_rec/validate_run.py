# Quick sanity check for a CaFE-Rec checkpoint.
# Loads the checkpoint, runs 6 cheap checks on a 100-row subset, and prints PASS/FAIL.
# Use this between Option A and Option D to catch silent failures (NaN, degenerate scores,
# bogus 1.0 metrics) before committing to a 7-hour full-data run.
#
# Exit code: 0 if all checks pass, 1 otherwise.
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from typing import List, Tuple

import numpy as np
import torch

from config import Config, CONFIG
from data.dataset import EvaluationDataset
from evaluation.rec_metrics import evaluate_recommendations
from models.cafe_rec import CaFeRec


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _pass(msg: str) -> Tuple[bool, str]:
    return True, f"{GREEN}PASS{RESET}  {msg}"


def _fail(msg: str) -> Tuple[bool, str]:
    return False, f"{RED}FAIL{RESET}  {msg}"


def _warn(msg: str) -> Tuple[bool, str]:
    return True, f"{YELLOW}WARN{RESET}  {msg}"


def load_artifacts(processed_dir: str):
    paths = {k: os.path.join(processed_dir, f"{k}.pkl") for k in ("train", "val", "test")}
    with open(paths["train"], "rb") as f:
        train_df = pickle.load(f)
    with open(paths["test"], "rb") as f:
        test_df = pickle.load(f)
    with open(os.path.join(processed_dir, "aspect_vocab.json"), "r") as f:
        vocab = json.load(f)
    return train_df, test_df, vocab


def build_and_load(checkpoint_path: str, train_df, test_df, vocab) -> CaFeRec:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Recover the config the model was trained with so dims match.
    cfg_dict = ckpt.get("config", {})
    cfg = CONFIG
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            cfg = cfg.__class__(**{**cfg.as_dict(), k: v})
    # Read the embedding matrix shapes directly from the checkpoint -- using
    # the dataframes' max indices fails when the checkpoint was trained on a
    # subsampled dataset (its embeddings may be 1-2 rows shorter than full).
    state = ckpt["model_state"]
    num_users = int(state["lightgcn.user_embedding.weight"].shape[0])
    num_items = int(state["lightgcn.item_embedding.weight"].shape[0])
    model = CaFeRec(cfg, num_users, num_items, vocab["aspects"])
    model.load_state_dict(state)
    model.to(torch.device(cfg.device))
    model.eval()
    # Filter train_df to indices the checkpoint knows about, so build_graph
    # doesn't reference out-of-range users/items.
    df = train_df[
        (train_df["user_idx"] < num_users) & (train_df["item_idx"] < num_items)
    ]
    model.build_graph(df)
    return model, cfg


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_weights_finite(model: CaFeRec) -> Tuple[bool, str]:
    """Are there any NaN or inf weights anywhere in the model?"""
    bad: List[str] = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
        if len(bad) >= 3:
            break
    if bad:
        return _fail(f"Model has NaN/inf weights in: {bad[:3]}")
    return _pass("All model weights are finite (no NaN/inf).")


def check_scores_nondegenerate(model: CaFeRec, train_df, n: int = 32) -> Tuple[bool, str]:
    """Do different (user, item) pairs produce *different* scores?
    A degenerate model (all-equal scores) is the silent failure mode that
    causes Recall@K = 1.0 even though the model learned nothing."""
    device = next(model.parameters()).device
    sample = train_df.sample(n=min(n, len(train_df)), random_state=0)
    u = torch.tensor(sample["user_idx"].values, dtype=torch.long, device=device)
    i = torch.tensor(sample["item_idx"].values, dtype=torch.long, device=device)
    with torch.no_grad():
        all_u, all_i = model.get_embeddings()
        u_emb, i_emb = all_u[u], all_i[i]
        selected, _ = model.aspect_selector(u_emb, i_emb, hard=True)
        scores = model.constrained_scorer(u_emb, i_emb, selected)
    s = scores.detach().cpu().numpy()
    if not np.isfinite(s).all():
        return _fail(f"Scores contain NaN/inf: min={np.nanmin(s):.4f} max={np.nanmax(s):.4f}")
    std = float(np.std(s))
    if std < 1e-6:
        return _fail(
            f"Scores are effectively constant (std={std:.2e}). "
            "Model is degenerate -- BPR collapsed or weights died."
        )
    return _pass(
        f"Scores vary across (user, item) pairs "
        f"(std={std:.3f}, min={s.min():.3f}, max={s.max():.3f})."
    )


def check_aspect_selector(model: CaFeRec, train_df, n: int = 16) -> Tuple[bool, str]:
    """Does the aspect selector actually pick a sparse set of aspects?
    If it picks all of them, or none, the explanation pathway is broken."""
    device = next(model.parameters()).device
    sample = train_df.sample(n=min(n, len(train_df)), random_state=1)
    u = torch.tensor(sample["user_idx"].values, dtype=torch.long, device=device)
    i = torch.tensor(sample["item_idx"].values, dtype=torch.long, device=device)
    with torch.no_grad():
        all_u, all_i = model.get_embeddings()
        selected, _ = model.aspect_selector(all_u[u], all_i[i], hard=True)
    active_per_row = selected.sum(dim=1).cpu().numpy()
    mean_active = float(active_per_row.mean())
    num_aspects = int(selected.shape[1])
    if mean_active < 0.5:
        return _fail(
            f"Aspect selector picks ~0 aspects/row (mean={mean_active:.2f}). "
            "L_sparse may have collapsed the gate."
        )
    if mean_active > num_aspects * 0.5:
        return _fail(
            f"Aspect selector picks too many aspects/row (mean={mean_active:.2f}/{num_aspects}). "
            "Hard selection isn't working."
        )
    return _pass(
        f"Aspect selector picks {mean_active:.1f}/{num_aspects} aspects/row "
        "(sparse, as designed)."
    )


def check_fidelity_drop(model: CaFeRec, train_df, n: int = 32) -> Tuple[bool, str]:
    """Counterfactual sanity: removing the selected aspects should drop the score.
    If it doesn't, the explanation is non-faithful and L_faith never trained."""
    device = next(model.parameters()).device
    sample = train_df.sample(n=min(n, len(train_df)), random_state=2)
    u = torch.tensor(sample["user_idx"].values, dtype=torch.long, device=device)
    i = torch.tensor(sample["item_idx"].values, dtype=torch.long, device=device)
    with torch.no_grad():
        all_u, all_i = model.get_embeddings()
        u_emb, i_emb = all_u[u], all_i[i]
        selected, _ = model.aspect_selector(u_emb, i_emb, hard=True)
        s_full = model.constrained_scorer(u_emb, i_emb, selected)
        # Mask out the selected aspects -> score should drop.
        s_masked = model.constrained_scorer(u_emb, i_emb, torch.zeros_like(selected))
    drop = (s_full - s_masked).cpu().numpy()
    mean_drop = float(drop.mean())
    if mean_drop <= 0:
        return _warn(
            f"Score does NOT drop when aspects removed (mean delta={mean_drop:+.3f}). "
            "Explanations may be non-faithful -- expected if model is still in warmup."
        )
    return _pass(
        f"Removing aspects drops score by {mean_drop:+.3f} on average (faithful)."
    )


def check_recall_realistic(model: CaFeRec, train_df, test_df, vocab, cfg: Config,
                           n_users: int = 100) -> Tuple[bool, str]:
    """Recall@10 on a 100-row sample should be in a sane range.
    Random scoring with 100 negatives -> Recall@10 ~ 10/101 ~ 0.10.
    A trained model should beat this. A bogus 1.0 -> eval bug or degenerate model."""
    # Use the model's embedding shape, not the dataframe max. If the model was
    # trained on a subsample, test rows can reference user/item indices the
    # embedding tables don't have -> CUDA index-out-of-bounds.
    model_num_users = int(model.lightgcn.user_embedding.weight.shape[0])
    model_num_items = int(model.lightgcn.item_embedding.weight.shape[0])

    test_in_range = test_df[
        (test_df["user_idx"] < model_num_users)
        & (test_df["item_idx"] < model_num_items)
    ]
    if len(test_in_range) == 0:
        return _warn(
            "No test rows fit inside the model's embedding range "
            f"({model_num_users:,} users x {model_num_items:,} items). "
            "Model was trained on a subset that excluded all test users/items."
        )
    sub_test = test_in_range.sample(
        n=min(n_users, len(test_in_range)), random_state=3
    ).reset_index(drop=True)

    train_in_range = train_df[
        (train_df["user_idx"] < model_num_users)
        & (train_df["item_idx"] < model_num_items)
    ]
    eval_ds = EvaluationDataset(
        sub_test, train_in_range, num_items=model_num_items,
        num_negatives=cfg.num_neg_samples, seed=42,
    )
    metrics = evaluate_recommendations(model, eval_ds, cfg, batch_size=32)
    r10 = float(metrics.get("Recall@10", 0.0))
    if r10 >= 0.99:
        return _fail(
            f"Recall@10 = {r10:.4f} -- bogus. "
            "Either model is degenerate or tie-breaking is still broken."
        )
    if r10 < 0.02:
        return _warn(
            f"Recall@10 = {r10:.4f} -- worse than random (0.10). "
            "Likely an undertrained model rather than a bug."
        )
    return _pass(
        f"Recall@10 = {r10:.4f} on 100-row sample (sane range; random would be ~0.10)."
    )


def check_explanation_generation(model: CaFeRec, train_df) -> Tuple[bool, str]:
    """Can the model generate a non-trivial explanation string?
    Useful smoke test for the Flan-T5 decoder path."""
    sample = train_df.sample(n=1, random_state=4).iloc[0]
    try:
        with torch.no_grad():
            text, top_aspects = model.generate_explanation(
                int(sample["user_idx"]), int(sample["item_idx"]), max_length=32
            )
    except Exception as e:
        return _fail(f"Explanation generation crashed: {type(e).__name__}: {e}")
    if not text or not text.strip():
        return _fail("Generated explanation is empty.")
    preview = text[:80].replace("\n", " ")
    return _pass(
        f"Generation works. Aspects={top_aspects[:3]}... Text: \"{preview}\""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Quick sanity check for a CaFE-Rec checkpoint.")
    ap.add_argument("--checkpoint", required=True, help="Path to a .pt file from train.py.")
    ap.add_argument("--processed_dir", default="data/processed")
    args = ap.parse_args()

    print(f"\n[validate] checkpoint = {args.checkpoint}")
    print(f"[validate] processed_dir = {args.processed_dir}\n")

    train_df, test_df, vocab = load_artifacts(args.processed_dir)
    model, cfg = build_and_load(args.checkpoint, train_df, test_df, vocab)
    print(f"[validate] model loaded on {cfg.device}, "
          f"{sum(p.numel() for p in model.parameters()):,} params\n")

    checks = [
        ("weights finite",       lambda: check_weights_finite(model)),
        ("scores non-degenerate", lambda: check_scores_nondegenerate(model, train_df)),
        ("aspect selector sparse", lambda: check_aspect_selector(model, train_df)),
        ("fidelity drop",         lambda: check_fidelity_drop(model, train_df)),
        ("recall realistic",      lambda: check_recall_realistic(model, train_df, test_df, vocab, cfg)),
        ("explanation gen works", lambda: check_explanation_generation(model, train_df)),
    ]

    n_pass = n_fail = 0
    for name, fn in checks:
        ok, msg = fn()
        print(f"  [{name:25s}]  {msg}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print(f"\n[validate] {n_pass} passed, {n_fail} failed.")
    if n_fail == 0:
        print(f"{GREEN}Checkpoint looks healthy. Safe to commit to Option D.{RESET}")
        return 0
    else:
        print(f"{RED}Do NOT commit to Option D. Debug failures above first.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
