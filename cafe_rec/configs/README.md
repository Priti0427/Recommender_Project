# Ablation configs

Override files for `train.py --config <path>`. Each one tweaks a small subset
of `Config` fields; everything else inherits from `config.py:CONFIG`.

The four configs map directly onto paper-table rows ("what happens if we
remove X from the joint loss?") so they need no further sweep over
hyperparameters — match the rest of the recipe (epochs, batch size, seed)
to whichever full-data CaFE-Rec run you compare against.

| Config                       | What it isolates                                                              |
|------------------------------|-------------------------------------------------------------------------------|
| `ablation_no_faith.json`     | Kills the counterfactual fidelity loss (the *novel* contribution).            |
| `ablation_no_sparse.json`    | Removes the L1 gate sparsity — selector becomes a dense soft mask.            |
| `ablation_warmup_only.json`  | Trains forever in warmup (only L_rec + L_text) — upper bound for "no gate".   |
| `ablation_faith_strong.json` | 4x lambda_faith — sensitivity check on the new loss.                          |

Each config writes to its own `save_dir` so checkpoints don't clobber the
main run.

## Reproducing the ablation rows

Run from the project root with the same dataset / sample size / seed as your
canonical CaFE-Rec full-data run. Example matching Option D (500k sample,
8 epochs, seed=0):

```bash
# 1. Main (paper table row "CaFE-Rec (ours)")
python -m train --dataset amazon_movies --sample_size 500000 \
    --num_epochs 8 --seed 0 --run_name cafe_rec_main

# 2. - L_faith
python -m train --dataset amazon_movies --sample_size 500000 \
    --num_epochs 8 --seed 0 --run_name cafe_rec_no_faith \
    --config configs/ablation_no_faith.json

# 3. - L_sparse
python -m train --dataset amazon_movies --sample_size 500000 \
    --num_epochs 8 --seed 0 --run_name cafe_rec_no_sparse \
    --config configs/ablation_no_sparse.json

# 4. Warmup-only (no L_faith, no L_sparse, never exits warmup)
python -m train --dataset amazon_movies --sample_size 500000 \
    --num_epochs 8 --seed 0 --run_name cafe_rec_warmup_only \
    --config configs/ablation_warmup_only.json

# 5. Sensitivity: 4x lambda_faith
python -m train --dataset amazon_movies --sample_size 500000 \
    --num_epochs 8 --seed 0 --run_name cafe_rec_faith_strong \
    --config configs/ablation_faith_strong.json
```

## What to report in the paper

For each ablation, report Recall@10 / NDCG@10 / BLEU-4 / Sufficiency /
Comprehensiveness / CF@10. The expected story:

- **No L_faith**: rec metrics roughly flat; CF@10 and Sufficiency collapse
  toward the warmup-only level — proves the new loss does the work.
- **No L_sparse**: selector spreads probability mass, explanations get
  generic; rec metrics may bump slightly because the scorer sees more signal.
- **Warmup-only**: the strongest "no-bottleneck" upper bound on Recall@10
  but worst Sufficiency / CF@10 — what XRec-style baselines look like.
- **Strong L_faith**: rec metrics may dip slightly; CF@10 climbs further —
  shows the rec–faith trade-off curve.
