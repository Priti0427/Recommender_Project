FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Project deps + API deps. Torch is the heavy one — pin to CPU wheels in the image
# for portability; for GPU/MPS run on the host instead.
COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api-requirements.txt
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /app/requirements.txt && \
    pip install -r /app/api-requirements.txt

COPY . /app

EXPOSE 8000

# Settings come from CAFE_* env vars; see api/settings.py.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
