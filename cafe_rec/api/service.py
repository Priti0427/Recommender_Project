"""Inference service. Owns the CaFE-Rec model and exposes high-level calls.

Loaded once at FastAPI startup; routers call into this object.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from config import Config
from models.cafe_rec import CaFeRec

from .settings import Settings


def _resolve_device(override: str | None) -> torch.device:
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class _Mappings:
    user2idx: Dict[str, int]
    item2idx: Dict[str, int]
    idx2user: Dict[int, str]
    idx2item: Dict[int, str]


class CaFERecService:
    """Singleton holding the loaded model + id mappings + aspect vocab."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = _resolve_device(settings.device)

        self._load_artifacts()
        self._build_and_load_model()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    def _load_artifacts(self) -> None:
        d = self.settings.processed_dir
        train_pkl = os.path.join(d, "train.pkl")
        mappings_json = os.path.join(d, "mappings.json")
        vocab_json = os.path.join(d, "aspect_vocab.json")

        for p in (train_pkl, mappings_json, vocab_json):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Required artifact missing: {p}. "
                    "Run preprocess.py and aspect_extract.py first."
                )

        with open(train_pkl, "rb") as f:
            self.train_df: pd.DataFrame = pickle.load(f)
        with open(mappings_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        with open(vocab_json, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.mappings = _Mappings(
            user2idx={k: int(v) for k, v in raw["user2idx"].items()},
            item2idx={k: int(v) for k, v in raw["item2idx"].items()},
            idx2user={int(k): v for k, v in raw["idx2user"].items()},
            idx2item={int(k): v for k, v in raw["idx2item"].items()},
        )

    def _build_and_load_model(self) -> None:
        ckpt_path = self.settings.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load to CPU first to avoid a known PyTorch-MPS bug copying bf16
        # tensors directly to the MPS device ("Unaligned blit request" assert).
        # The model is moved to self.device after load_state_dict below.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # train.py saves config as a dict (config.as_dict()); rebuild a Config and
        # copy over any fields the dataclass knows about. Fall back to defaults
        # when the checkpoint is missing the config block entirely.
        raw_config = ckpt.get("config")
        config = Config()
        if isinstance(raw_config, dict):
            for k, v in raw_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        elif raw_config is not None:
            config = raw_config  # already a Config dataclass
        config.device = str(self.device)

        # Derive the embedding-table sizes from the checkpoint itself, NOT from
        # the full mappings -- the checkpoint may have been trained on an
        # active-user subset (smaller index space) and the model must match it.
        state = ckpt["model_state"]
        num_users = int(state["lightgcn.user_embedding.weight"].shape[0])
        num_items = int(state["lightgcn.item_embedding.weight"].shape[0])
        # Save so the @property accessors return the right size too.
        self._num_users = num_users
        self._num_items = num_items

        model = CaFeRec(
            config=config,
            num_users=num_users,
            num_items=num_items,
            aspect_vocabulary=self.vocab["aspects"],
        )
        model.load_state_dict(ckpt["model_state"], strict=False)
        model.to(self.device)

        # Filter train_df to the checkpoint's index range so build_graph doesn't
        # reference users/items the embedding tables don't have.
        train_df_in_range = self.train_df[
            (self.train_df["user_idx"] < num_users)
            & (self.train_df["item_idx"] < num_items)
        ]
        model.build_graph(train_df_in_range)
        model.eval()

        # Pre-compute a stable sample of trained users for the UI dropdown.
        # We pick from the most-active users in the in-range train set so the
        # recommendations are meaningful (sparse users would give random-looking
        # output). 200 cached samples is enough to give the UI a randomized
        # subset per request without re-scanning the dataframe.
        try:
            counts = (
                train_df_in_range.groupby("user_idx").size()
                .sort_values(ascending=False)
            )
            top_idx = list(counts.head(200).index)
            self._sample_users = [
                {"user_idx": int(i),
                 "user_id": self.mappings.idx2user.get(int(i), str(int(i))),
                 "n_interactions": int(counts.loc[int(i)])}
                for i in top_idx
            ]
        except Exception:
            self._sample_users = []

        self.model = model
        self.config = config

        # Cache LightGCN propagated embeddings — they're static for inference.
        with torch.no_grad():
            u_emb, i_emb = self.model.get_embeddings()
        self._user_emb = u_emb.detach()
        self._item_emb = i_emb.detach()

    # ------------------------------------------------------------------
    # Public meta
    # ------------------------------------------------------------------
    @property
    def num_users(self) -> int:
        # Report the model's embedding-table size (may be smaller than mappings
        # if the checkpoint was trained on a subset).
        return getattr(self, "_num_users", len(self.mappings.user2idx))

    @property
    def num_items(self) -> int:
        return getattr(self, "_num_items", len(self.mappings.item2idx))

    @property
    def num_aspects(self) -> int:
        return len(self.vocab["aspects"])

    def aspects(self) -> List[str]:
        return list(self.vocab["aspects"])

    # ------------------------------------------------------------------
    # ID lookups
    # ------------------------------------------------------------------
    def user_to_idx(self, user_id: str) -> int:
        if user_id in self.mappings.user2idx:
            return self.mappings.user2idx[user_id]
        if user_id.isdigit() and int(user_id) in self.mappings.idx2user:
            return int(user_id)
        raise KeyError(f"unknown user_id: {user_id}")

    def item_to_idx(self, item_id: str) -> int:
        if item_id in self.mappings.item2idx:
            return self.mappings.item2idx[item_id]
        if item_id.isdigit() and int(item_id) in self.mappings.idx2item:
            return int(item_id)
        raise KeyError(f"unknown item_id: {item_id}")

    def idx_to_item(self, idx: int) -> str:
        return self.mappings.idx2item.get(int(idx), str(idx))

    def idx_to_user(self, idx: int) -> str:
        return self.mappings.idx2user.get(int(idx), str(idx))

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score_batch(
        self, user_idx: int, item_idxs: torch.Tensor
    ) -> torch.Tensor:
        """ConstrainedScorer over (one user) x (many items). Returns [N]."""
        u_emb = self._user_emb[user_idx].unsqueeze(0).expand(item_idxs.size(0), -1)
        i_emb = self._item_emb[item_idxs]
        sel, _ = self.model.aspect_selector(u_emb, i_emb, hard=True)
        return self.model.constrained_scorer(u_emb, i_emb, sel).view(-1)

    @torch.no_grad()
    def recommend(self, user_id: str, k: int) -> List[Tuple[int, float]]:
        """Return top-k (item_idx, score) for the user, excluding training positives."""
        u_idx = self.user_to_idx(user_id)
        all_items = torch.arange(self.num_items, device=self.device)

        # Score in chunks to stay friendly on MPS.
        chunk = 4096
        scores = torch.empty(self.num_items, device=self.device)
        for start in range(0, self.num_items, chunk):
            end = min(start + chunk, self.num_items)
            scores[start:end] = self._score_batch(u_idx, all_items[start:end])

        # Mask items the user has already interacted with in train.
        seen = self.train_df.loc[
            self.train_df["user_idx"] == u_idx, "item_idx"
        ].to_numpy()
        if seen.size:
            scores[torch.as_tensor(seen, dtype=torch.long, device=self.device)] = -float("inf")

        topv, topi = torch.topk(scores, k=min(k, self.num_items))
        return list(zip(topi.cpu().tolist(), topv.cpu().tolist()))

    @torch.no_grad()
    def explain(
        self, user_id: str, item_id: str,
        max_length: int | None = None,
        num_beams: int | None = None,
    ) -> Dict:
        """Score + selected aspects + generated explanation for one (u, i)."""
        u_idx = self.user_to_idx(user_id)
        i_idx = self.item_to_idx(item_id)

        u_emb = self._user_emb[u_idx].unsqueeze(0)
        i_emb = self._item_emb[i_idx].unsqueeze(0)

        selected, logits = self.model.aspect_selector(u_emb, i_emb, hard=True)
        score = self.model.constrained_scorer(u_emb, i_emb, selected).view(-1).item()

        k = int(self.config.num_selected_aspects)
        top_idx = selected.topk(min(k, selected.size(-1)), dim=-1).indices[0].cpu().tolist()
        aspect_names = [self.vocab["aspects"][j] for j in top_idx]
        aspect_weights = [float(selected[0, j].item()) for j in top_idx]

        text, _ = self.model.generate_explanation(
            user_id=u_idx,
            item_id=i_idx,
            max_length=int(max_length or self.settings.generation_max_length),
            num_beams=int(num_beams or self.settings.generation_num_beams),
        )

        return {
            "score": float(score),
            "selected_aspects": aspect_names,
            "aspect_weights": aspect_weights,
            "explanation": text,
        }

    @torch.no_grad()
    def counterfactual(
        self, user_id: str, item_id: str, num_candidates: int, top_k: int,
    ) -> Dict:
        """Single-pair fidelity: score factual vs. counterfactual + rank-flip test."""
        u_idx = self.user_to_idx(user_id)
        i_idx = self.item_to_idx(item_id)

        u_emb = self._user_emb[u_idx].unsqueeze(0)
        i_emb = self._item_emb[i_idx].unsqueeze(0)

        selected, _ = self.model.aspect_selector(u_emb, i_emb, hard=True)
        k_aspects = int(self.config.num_selected_aspects)
        top_idx = selected.topk(min(k_aspects, selected.size(-1)), dim=-1).indices
        expl_mask = torch.zeros_like(selected)
        expl_mask.scatter_(1, top_idx, 1.0)

        factual = self.model.constrained_scorer(u_emb, i_emb, selected).view(-1).item()
        cf = self.model.constrained_scorer(
            u_emb, i_emb, selected * (1.0 - expl_mask)
        ).view(-1).item()
        only = self.model.constrained_scorer(
            u_emb, i_emb, selected * expl_mask
        ).view(-1).item()

        sufficiency = 1.0 - abs(factual - only) / (abs(factual) + 1e-8)
        comprehensiveness = factual - cf

        # Rank-flip test: sample N negatives and check whether the item drops
        # out of the top-K when we zero its explanation aspects.
        rng = np.random.RandomState(42 + u_idx)
        seen = set(
            self.train_df.loc[
                self.train_df["user_idx"] == u_idx, "item_idx"
            ].astype(int).tolist()
        )
        seen.add(i_idx)
        negs: List[int] = []
        while len(negs) < num_candidates:
            cand = rng.randint(0, self.num_items, size=num_candidates * 2)
            for c in cand:
                c = int(c)
                if c in seen:
                    continue
                negs.append(c)
                seen.add(c)
                if len(negs) == num_candidates:
                    break

        cands = torch.tensor([i_idx] + negs, dtype=torch.long, device=self.device)
        Nc = cands.size(0)
        u_exp = self._user_emb[u_idx].unsqueeze(0).expand(Nc, -1)
        i_exp = self._item_emb[cands]
        sel_all, _ = self.model.aspect_selector(u_exp, i_exp, hard=True)
        fact_scores = self.model.constrained_scorer(u_exp, i_exp, sel_all).view(-1)
        cf_scores = self.model.constrained_scorer(
            u_exp, i_exp,
            sel_all * (1.0 - expl_mask.expand(Nc, -1)),
        ).view(-1)

        fact_rank = int((fact_scores > fact_scores[0]).sum().item()) + 1
        cf_rank = int((cf_scores > cf_scores[0]).sum().item()) + 1
        flipped = fact_rank <= top_k and cf_rank > top_k

        aspect_names = [self.vocab["aspects"][j] for j in top_idx[0].cpu().tolist()]
        return {
            "selected_aspects": aspect_names,
            "factual_score": float(factual),
            "counterfactual_score": float(cf),
            "delta": float(factual - cf),
            "sufficiency": float(sufficiency),
            "comprehensiveness": float(comprehensiveness),
            "factual_rank": fact_rank,
            "counterfactual_rank": cf_rank,
            "flipped_out_of_top_k": bool(flipped),
            "top_k": int(top_k),
        }
