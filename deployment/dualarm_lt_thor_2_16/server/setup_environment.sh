#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_RUNTIME_SITE="${STARVLA_CUDA_RUNTIME_SITE:-/home/test/Public/zpf/venvs/lerobot061/lib/python3.12/site-packages}"
test -d "$CUDA_RUNTIME_SITE/torch"
python3 -m venv --without-pip "$HERE/.venv"
printf '%s\n' "$CUDA_RUNTIME_SITE" > "$HERE/.venv/lib/python3.12/site-packages/thor_cuda_runtime.pth"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  PIP_CONFIG_FILE=/dev/null "$HERE/.venv/bin/python" -m pip install \
  --proxy '' --index-url https://pypi.org/simple --ignore-installed --no-deps \
  transformers==4.57.0 huggingface-hub==0.34.4 omegaconf==2.3.0 antlr4-python3-runtime==4.9.3
"$HERE/.venv/bin/python" -c 'import torch, torchvision, transformers, omegaconf; print(torch.__version__, torchvision.__version__, transformers.__version__, torch.cuda.is_available())'
