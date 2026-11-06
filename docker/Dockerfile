FROM python:3.12-slim-bookworm

# build-essential: torch.compile needs a C++ toolchain. git: the SDK is installed from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# One resolution pass, every version fixed by docker/constraints.txt. This matters: `tensordict>=0.15` requires
# torch>=2.11, so installing Idol on its own drags in a second, different torch and
# silently replaces the CUDA build. 0.12.3 has no torch floor and a cp312 wheel.
# The cu128 build covers sm_86 (RTX 3080) through sm_120 (RTX 5090); torchvision
# 0.24.1 is the release that requires torch==2.9.1 exactly.
# The cache mount means a retried build reuses these ~4 GB of wheels instead of
# downloading them again.
COPY docker/constraints.txt /tmp/constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -c /tmp/constraints.txt \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" "tensordict==0.12.3" \
        "idol @ git+https://github.com/speridlabs/idol-sdk@4702a7f00850e34b6e83a2170cc123c3e9570744" \
        "mcp>=2.3" fastapi uvicorn requests \
    && python -c "import torch, idol, diffusers, transformers; from tensordict import TensorDict; \
assert torch.__version__ == '2.9.1+cu128', 'wrong torch: ' + torch.__version__; \
print('stack ok: torch', torch.__version__, '| idol', idol.__version__)"

ENV PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    HF_HUB_OFFLINE=1 \
    TORCHINDUCTOR_CACHE_DIR=/data/.cache/inductor \
    IDOL_DATA=/data \
    IDOL_MODEL_DIR=/data/idol-local \
    IDOL_PORT=8043

WORKDIR /app
COPY server/ /app/server/
COPY web/ /app/web/
COPY tools/ /app/tools/

EXPOSE 8043
CMD ["python", "/app/server/app.py"]
