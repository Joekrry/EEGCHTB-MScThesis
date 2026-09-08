# torch==2.12.1+cu130 pulls its own CUDA 13 runtime libs from the cu130 wheel index
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies before copying source so code edits don't bust the layer cache
COPY requirements.txt pyproject.toml ./
COPY eegchtb ./eegchtb
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY . .

# Datasets (BCI_IV_2A_DIR / BCI_IV_2B_DIR) and run outputs are expected to be
# bind-mounted at container run time, not baked into the image.
ENTRYPOINT ["eegchtb-run"]
