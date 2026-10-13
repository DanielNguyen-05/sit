#!/bin/bash
# One-time preparation for eval_fid.py. Run on the LOGIN node (it has internet; GPU nodes may not):
#
#   cd /media/ltnghia33/sit/v2 && bash prepare_fid.sh
#
# 1. installs pytorch-fid (Inception-v3 for FID) into the "sit" environment
# 2. caches the Inception weights (~92MB) and the SD-VAE decoder in your home folder
# 3. downloads OpenAI's ADM ImageNet-256 reference batch (~2GB) to the login node's local /tmp,
#    keeps only its mu / sigma (~34MB) in $OUT, and deletes the big file, so /media barely grows.
set -eo pipefail

OUT=${OUT:-/media/ltnghia33/sit/fid_ref}
DL_DIR=${DL_DIR:-/tmp/${USER}_fidref}
URL=https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
STATS="$OUT/VIRTUAL_imagenet256_labeled_stats.npz"

source /media/ltnghia33/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sit}"

echo "== 1/3 pytorch-fid"
pip install -q "pytorch-fid==0.3.0" scipy

echo "== 2/3 Inception weights + VAE (cached in your home folder)"
python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3]); print('Inception OK')"
# Resolve the VAE to a real folder (config.json + weights) and record it, so GPU jobs load it from
# disk by path and never ask the Hugging Face Hub anything (offline lookups failed on the GPU nodes).
mkdir -p "$OUT"
VAE_DIR=$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("stabilityai/sd-vae-ft-ema",
                        allow_patterns=["config.json", "diffusion_pytorch_model.safetensors"]))
PY
)
case "$VAE_DIR" in
    /media/*) ;;                                   # on the shared disk: GPU nodes can read it
    *) echo "VAE cache is not on the shared disk ($VAE_DIR); copying it to $OUT/sd-vae-ft-ema"
       rm -rf "$OUT/sd-vae-ft-ema"; cp -rL "$VAE_DIR" "$OUT/sd-vae-ft-ema"; VAE_DIR="$OUT/sd-vae-ft-ema" ;;
esac
python -c "import sys; from diffusers.models import AutoencoderKL; AutoencoderKL.from_pretrained(sys.argv[1]); print('VAE OK:', sys.argv[1])" "$VAE_DIR"
echo "$VAE_DIR" > "$OUT/vae_path.txt"

echo "== 3/3 reference statistics"
if [ -f "$STATS" ]; then
    echo "already there: $STATS"
else
    mkdir -p "$OUT" "$DL_DIR"
    AVAIL_GB=$(df -Pk "$DL_DIR" | awk 'NR==2{print int($4/1024/1024)}')
    echo "free space in $DL_DIR: ${AVAIL_GB}GB"
    if [ "$AVAIL_GB" -lt 3 ]; then
        echo "ERROR: need ~3GB free in $DL_DIR. Rerun with DL_DIR=/some/folder/with/space bash prepare_fid.sh"; exit 1
    fi
    wget -c -O "$DL_DIR/ref.npz" "$URL"
    python - "$DL_DIR/ref.npz" "$STATS" <<'PY'
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
echo "Done. Submit FID jobs with: sbatch fid.sh <checkpoint> [<checkpoint> ...]"
