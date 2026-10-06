# LightNorm journal runtime: Python 3.10, torch 2.1.0, CUDA 12.1 (A100/H100).
# Build:  docker build -f docker/Dockerfile -t lightnorm-journal:cu121 .
# The host needs an NVIDIA driver >= 525 (CUDA 12.x minor-version compatibility).
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates unzip rsync tmux htop procps less vim-tiny \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-timemixerpp.txt /tmp/
RUN pip install -r /tmp/requirements.txt \
    && pip install --no-deps -r /tmp/requirements-timemixerpp.txt \
    && python -c "import torch, pytorch_wavelets, statsmodels, pypots; assert torch.__version__.startswith('2.1.0')"

WORKDIR /workspace
