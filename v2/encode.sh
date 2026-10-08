#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#
# Builds latents.npy + labels.npy from ImageNet-1k (Hugging Face parquet shards).
# Submit from the folder that contains encode_latents.py, after `mkdir -p logs`:
#
#   sbatch encode.sh            # GPU node has internet: download + encode + delete, shard by shard
#   WAIT=1 sbatch encode.sh     # GPU node has NO internet: also run, on the login node,
#                               #   python download_shards.py --src /media/ltnghia33/imagenet_parquet
#
# Interrupted or timed out? Submit the same command again; it continues where it stopped.

set -eo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
OUT=${OUT:-/media/ltnghia33/sit}                    # where latents.npy / labels.npy go
SRC=${SRC:-/media/ltnghia33/imagenet_parquet}       # scratch folder for the parquet shards
CONDA_ENV=${CONDA_ENV:-sit}

source /media/ltnghia33/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"

if [ "${WAIT:-0}" = "1" ]; then
    FETCH="--wait"
    export HF_HUB_OFFLINE=1      # the VAE must already be in the Hugging Face cache (see step 3)
else
    FETCH="--hf-download"
fi

echo "Node: $(hostname)   Start: $(date)   mode: $FETCH"
df -h "$(dirname "$OUT")" | tail -1
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-12} - 2 ))

python -u encode_latents.py --out "$OUT" --src "$SRC" $FETCH --delete-shards --num-workers "$WORKERS" "$@"

echo "End: $(date)"
