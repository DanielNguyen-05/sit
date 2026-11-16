#!/bin/bash
# One-time preparation for eval_fid.py. Run on the LOGIN node (it has internet; GPU nodes may not),
# from the repository root:
#
#   cd /media/ltnghia33/sit && bash scripts/prepare_fid.sh
#
# Everything goes to fid_ref/ (FID_REF in env.sh), not ~/.cache, which the server cleans up:
# 1. installs pytorch-fid (Inception-v3 for FID) into the conda environment
# 2. Inception weights (~92MB) -> fid_ref/torch/, SD-VAE decoder -> fid_ref/sd-vae-ft-ema/
# 3. downloads OpenAI's ADM ImageNet-256 reference batch (~2GB) to local /tmp, keeps only its
#    mu / sigma (~34MB) in fid_ref/, and deletes the big file.
set -eo pipefail

_r=${SIT_ROOT:-$PWD}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"

OUT="$SIT_FID_REF"
DL_DIR=${DL_DIR:-/tmp/${USER}_fidref}
URL=https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

activate_env
mkdir -p "$OUT"

echo "== 1/3 pytorch-fid"
pip install -q "pytorch-fid==0.3.0" scipy

echo "== 2/3 Inception weights + VAE (both into $OUT)"
python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3]); print('Inception OK')"
# Download the VAE as plain files into our own folder (not the Hugging Face cache), so GPU jobs
# load it by path and never ask the Hub anything.
if [ -f "$VAE_DIR/config.json" ] && [ -f "$VAE_DIR/diffusion_pytorch_model.safetensors" ]; then
    echo "VAE already there: $VAE_DIR"
else
    python - "$VAE_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/sd-vae-ft-ema", local_dir=sys.argv[1],
                  allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"])
PY
fi
python -c "import sys; from diffusers.models import AutoencoderKL; AutoencoderKL.from_pretrained(sys.argv[1]); print('VAE OK:', sys.argv[1])" "$VAE_DIR"

echo "== 3/3 reference statistics"
if [ -f "$REF_STATS" ]; then
    echo "already there: $REF_STATS"
else
    mkdir -p "$DL_DIR"
    AVAIL_GB=$(df -Pk "$DL_DIR" | awk 'NR==2{print int($4/1024/1024)}')
    echo "free space in $DL_DIR: ${AVAIL_GB}GB"
    if [ "$AVAIL_GB" -lt 3 ]; then
        echo "ERROR: need ~3GB free in $DL_DIR. Rerun with DL_DIR=/some/folder/with/space bash scripts/prepare_fid.sh"; exit 1
    fi
    wget -c -O "$DL_DIR/ref.npz" "$URL"
    python - "$DL_DIR/ref.npz" "$REF_STATS" <<'PY'
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
ref = np.load(src)
mu, sigma = ref["mu"], ref["sigma"]
assert mu.shape == (2048,) and sigma.shape == (2048, 2048), (mu.shape, sigma.shape)
np.savez(dst, mu=mu, sigma=sigma)
print(f"saved {dst}: mu {mu.shape}, sigma {sigma.shape}")
PY
    rm -f "$DL_DIR/ref.npz"
fi
require_fid_ref
echo "Done. Submit FID jobs with: sbatch scripts/fid.sh <checkpoint> [<checkpoint> ...]"
