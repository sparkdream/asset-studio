# Project environment: source it from a shell, it only affects that shell.
#
#     source env.sh
#
# Machine-specific settings (e.g. HF_HOME) go in env.local.sh, which git ignores and this file loads.

# TRELLIS.2 backends that work without the CUDA-only extensions (Intel GPUs)
export SPARSE_CONV_BACKEND=torch
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa

if [ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/env.local.sh" ]; then
    . "$(dirname "${BASH_SOURCE[0]:-$0}")/env.local.sh"
fi
