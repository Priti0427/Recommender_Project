# Baselines

Each runner trains a baseline on the same processed splits as CaFE-Rec and
writes two files into `--out`:

```
<name>.json              {dataset: {"Recall@10": ..., "NDCG@10": ..., ...}}
<name>_per_example.npz   {"Recall@K": [N], "NDCG@K": [N]}  (for Wilcoxon)
```

`evaluate.py` already understands this layout via `load_baseline_results`.
The per-example NPZ enables paired Wilcoxon signed-rank tests against
CaFE-Rec's per-example arrays (see `evaluate.py:run_significance_tests`).

## Available baselines

| Script               | What it is                              | Has text head? | Runtime estimate (Colab T4)        |
|----------------------|-----------------------------------------|----------------|------------------------------------|
| `run_random.py`      | Uniform-random scorer                   | no             | seconds                            |
| `run_popularity.py`  | Score = `log1p(train_item_count)`       | no             | seconds                            |
| `run_bprmf.py`       | BPR-MF (same dim as CaFE-Rec)           | no             | ~10 min for 10 ep on 100k          |
| `run_peter.py`       | PETER — small transformer, joint LM+rating | yes        | ~30 min for 8 ep on 100k           |
| `run_pepler.py`      | PEPLER — Flan-T5-small + soft prompts   | yes            | ~45 min for 5 ep on 100k           |
| `run_xrec.py`        | XRec — LightGCN + Flan-T5-base adapter  | yes            | ~2 h for 3 ep on 100k              |

The three "explainable" baselines (`peter`, `pepler`, `xrec`) leave
`sufficiency`, `comprehensiveness`, and `CF@10` set to `None` so the
comparison table prints **N/A** rather than 0 — none of these models trains
for faithfulness, so reporting 0 would be misleading.

## Usage

```bash
# Floor baselines (no training; instant)
python -m baselines.run_random      --dataset amazon_movies \
    --processed_dir cafe_rec/data/processed --out results/baselines
python -m baselines.run_popularity  --dataset amazon_movies \
    --processed_dir cafe_rec/data/processed --out results/baselines

# CF baseline (no text head)
python -m baselines.run_bprmf       --dataset amazon_movies \
    --processed_dir cafe_rec/data/processed --out results/baselines \
    --epochs 10 --train_batch_size 1024

# Explainable baselines
python -m baselines.run_peter       --dataset amazon_movies \
    --processed_dir cafe_rec/data/processed --out results/baselines \
    --epochs 8 --train_batch_size 64

python -m baselines.run_pepler      --dataset amazon_movies \
    --processed_dir cafe_rec/data/processed --out results/baselines \
    --epochs 5 --train_batch_size 32 --lm_name google/flan-t5-small

python -m baselines.run_xrec        --dataset amazon_movies \
    --processed_dir cafe_rec/data/processed --out results/baselines \
    --epochs 3 --bpr_pretrain_epochs 3 \
    --text_batch_size 16 --bpr_batch_size 1024 \
    --lm_name google/flan-t5-base
```

For a fast smoke test, add `--sample_train 50000 --sample_test 2000`.

## Fairness of the comparison

The harness pins five things to keep the comparison apples-to-apples:

1. **Same processed splits** — train/val/test pickles produced by
   `data/preprocess.py`. No baseline ever sees test reviews during training.
2. **Same evaluation slate** — `(num_negatives=100, seed=42)` so the
   per-example metric arrays are paired and Wilcoxon is valid.
3. **Same metric implementations** — `evaluation/rec_metrics.py` (with the
   recent NaN/tie-breaking fix) and `evaluation/text_metrics.py`.
4. **Same LM family** for explainable baselines (Flan-T5). Where the upstream
   paper used a different backbone (GPT-2 for PEPLER, Llama-2 for XRec),
   each script's top-of-file comment documents the deviation and reasoning.
5. **Same embedding capacity** on the rec side: `config.embedding_dim` for
   anything that has user/item embeddings.

## Vendoring upstream code

If you want a paper-faithful reimplementation rather than the in-file ones,
each script honors a `<NAME>_USE_UPSTREAM` env var. Drop the upstream repo
at `baselines/<name>_src/` and run with the env var set; the runner imports
that module's model class and skips the in-file fallback. The data loading,
eval, and output paths stay the same.

```bash
git clone https://github.com/lileipisces/PETER  cafe_rec/baselines/peter_src
PETER_USE_UPSTREAM=1 python -m baselines.run_peter --dataset amazon_movies ...
```
