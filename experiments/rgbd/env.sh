#!/usr/bin/env bash
mountpoint -q /root/data || { echo "Data mount missing" >&2; return 1; }
export HF_HOME=/root/data/huggingface HF_HUB_CACHE=/root/data/huggingface/hub
export UV_CACHE_DIR=/root/data/uv/cache UV_PYTHON_INSTALL_DIR=/root/data/uv/python
export TORCH_HOME=/root/data/torch-cache XDG_CACHE_HOME=/root/data/.cache
export TMPDIR=/root/data/tmp PIP_CACHE_DIR=/root/data/.cache/pip
export PYTHONPATH=/root/data/projects/starVLA-rgbd:/root/data/projects/lingbot-depth:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export LIBERO_CONFIG_PATH=/root/data/.cache/libero-rgbd
export NUMBA_CACHE_DIR=/root/data/.cache/numba-rgbd
export PYTHONPATH=/root/data/projects/LIBERO-rgbd:$PYTHONPATH
