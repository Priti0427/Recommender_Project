from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAFE_", env_file=".env", extra="ignore")

    checkpoint_path: str = Field(
        default=str(REPO_ROOT / "checkpoints" / "cafe_rec_amazon_movies_best_e01_r0.0938.pt"),
        description="Path to a trained CaFE-Rec checkpoint (.pt).",
    )
    processed_dir: str = Field(
        default=str(REPO_ROOT / "data" / "processed"),
        description="Directory containing train.pkl, mappings.json, aspect_vocab.json.",
    )
    device: str | None = Field(
        default=None,
        description="Override torch device. None -> auto (mps/cuda/cpu).",
    )
    default_top_k: int = 10
    max_top_k: int = 50
    cf_num_negatives: int = 100
    generation_max_length: int = 64
    generation_num_beams: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


def get_settings() -> Settings:
    return Settings()
