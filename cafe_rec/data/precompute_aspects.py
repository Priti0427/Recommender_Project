# One-shot: load each split pickle, attach aspect_labels, save back in place.
# Run once after preprocessing -- removes the ~40-min reload tax on every future train.py run.
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import pandas as pd

# Ensure we can import from the cafe_rec package root when invoked as a script.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from data.aspect_extract import add_aspect_labels_to_dataset  # noqa: E402


def precompute(processed_dir: str, splits=("train", "val", "test")) -> None:
    vocab_path = os.path.join(processed_dir, "aspect_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"{vocab_path} -- run data.aspect_extract first.")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    for name in splits:
        path = os.path.join(processed_dir, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        t0 = time.time()
        with open(path, "rb") as f:
            df: pd.DataFrame = pickle.load(f)
        if "aspect_labels" in df.columns:
            print(f"[skip] {name}.pkl already has aspect_labels "
                  f"({len(df):,} rows)")
            continue
        print(f"[{name}] encoding aspect labels for {len(df):,} rows...")
        df = add_aspect_labels_to_dataset(df, vocab)
        # Atomic-ish replace via tmp.
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        print(f"[{name}] saved {path} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    args = ap.parse_args()
    precompute(args.processed_dir)
