# CaFE-Rec API

FastAPI service that loads a trained CaFE-Rec checkpoint and exposes inference
endpoints for top-K recommendations, natural-language explanations, and
counterfactual fidelity diagnostics.

## Run

From the `cafe_rec/` directory (so imports of `config`, `models`, `data` resolve):

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

Open http://localhost:8000/ for the demo UI, or http://localhost:8000/docs for Swagger.

## Configuration

All settings are env-overridable (prefix `CAFE_`):

| Var                       | Default                                                         |
|---------------------------|-----------------------------------------------------------------|
| `CAFE_CHECKPOINT_PATH`    | `./checkpoints/cafe_rec_amazon_movies_best_e01_r0.0938.pt`      |
| `CAFE_PROCESSED_DIR`      | `./data/processed`                                              |
| `CAFE_DEVICE`             | auto (`mps` / `cuda` / `cpu`)                                   |
| `CAFE_DEFAULT_TOP_K`      | `10`                                                            |
| `CAFE_GENERATION_NUM_BEAMS` | `4`                                                           |

## Endpoints

| Method | Path                              | Purpose                                        |
|--------|-----------------------------------|------------------------------------------------|
| GET    | `/`                               | single-page demo UI (Tailwind, vanilla JS)     |
| GET    | `/health`                         | liveness + model metadata                      |
| GET    | `/aspects`                        | aspect vocabulary                              |
| GET    | `/users/{user_id}/recommend?k=10` | top-K items for a user                         |
| POST   | `/explain`                        | aspects + generated text + score for one pair  |
| POST   | `/counterfactual`                 | sufficiency / comprehensiveness / rank-flip    |

The demo UI lets you enter a user id, see top-K recs as cards, and click any
card to load the generated explanation alongside its counterfactual fidelity
panel — including the rank-flip indicator that operationalizes the paper's
core claim.

`user_id` and `item_id` accept either the original raw id (Amazon ASIN, etc.)
or the integer index used internally.

## Example

```bash
# top-10 for a known user
curl 'http://localhost:8000/users/AE2222FRPDMNOMYOMCWIANTXP7UQ/recommend?k=10'

# explain a (user, item) pair
curl -X POST http://localhost:8000/explain \
  -H 'content-type: application/json' \
  -d '{"user_id": "AE2222FRPDMNOMYOMCWIANTXP7UQ", "item_id": "000014357X"}'

# counterfactual diagnostics on the same pair
curl -X POST http://localhost:8000/counterfactual \
  -H 'content-type: application/json' \
  -d '{"user_id": "AE2222FRPDMNOMYOMCWIANTXP7UQ", "item_id": "000014357X", "top_k": 10}'
```

## Docker

```bash
docker build -f api/Dockerfile -t cafe-rec-api .
docker run -p 8000:8000 \
  -e CAFE_CHECKPOINT_PATH=/app/checkpoints/cafe_rec_amazon_movies_best_e01_r0.0938.pt \
  -v "$PWD/checkpoints:/app/checkpoints" \
  -v "$PWD/data/processed:/app/data/processed" \
  cafe-rec-api
```

The image installs CPU-only PyTorch. For GPU/MPS, run uvicorn on the host instead.
