# CaFE-Rec: Counterfactually Faithful Explanation Recommender

An explainable recommender system that produces natural-language explanations **causally tied** to the recommendation — not post-hoc rationalizations. Built for DSCI 641 (Drexel) and targeting RecSys 2026.

## What's inside

- [`cafe_rec/`](cafe_rec/) — the source package: LightGCN backbone, Gumbel-Softmax aspect selector, constrained scorer, Flan-T5-Base text adapter, training/eval scripts, baselines, REST API and demo UI.
- [`CaFE_Rec_Colab.ipynb`](CaFE_Rec_Colab.ipynb) — end-to-end Colab notebook for running the pipeline.
- [`FINAL_REPORT.md`](FINAL_REPORT.md) — project writeup.
- [`report_assets/`](report_assets/) — UI screenshots.

See [`cafe_rec/README.md`](cafe_rec/README.md) for setup, dataset download, training, evaluation and API instructions.

## Quick start

```bash
cd cafe_rec
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Model summary

Four modules trained end-to-end:

1. **LightGCN** — graph-aware user/item embeddings
2. **AspectSelector** — 2-layer MLP + Gumbel-Softmax picks 3–5 aspects from a review-mined vocabulary
3. **ConstrainedScorer** — MLP that scores recommendations using *only* the selected aspects
4. **Flan-T5-Base adapter** — generates text from the same aspect bottleneck

Jointly optimized over BPR ranking, text cross-entropy, counterfactual fidelity, and L1 sparsity.
