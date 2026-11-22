#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --partition=batch
#SBATCH --output=logs/prepare/%x_%j.out
#SBATCH --error=logs/prepare/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#
# Builds data/latents.npy + data/labels.npy from ImageNet-1k (Hugging Face parquet shards).
# Only needed once; submit from the repository root:
#
#   sbatch scripts/encode.sh            # GPU node has internet: download + encode + delete, shard by shard
#   WAIT=1 sbatch scripts/encode.sh     # GPU node has NO internet: also run, on the login node,
#                                       #   python tools/download_shards.py --src "$PARQUET"
#
# Interrupted or timed out? Submit the same command again; it continues where it stopped.
# Folders: DATA (output) and PARQUET (scratch for the shards) in scripts/env.sh.

set -eo pipefail

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"

activate_env
cd "$SIT_ROOT"
mkdir -p "$SIT_DATA" "$PARQUET"

if [ "${WAIT:-0}" = "1" ]; then
    FETCH="--wait"
    export HF_HUB_OFFLINE=1      # the VAE must already be local (scripts/prepare_fid.sh)
else
    FETCH="--hf-download"
fi
VAE_ARG=()
[ -f "$VAE_DIR/config.json" ] && VAE_ARG=(--vae "$VAE_DIR")

echo "Node: $(hostname)   Start: $(date)   mode: $FETCH"
df -h "$SIT_DATA" | tail -1
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-12} - 2 ))

python -u tools/encode_latents.py --out "$SIT_DATA" --src "$PARQUET" $FETCH --delete-shards \
    --num-workers "$WORKERS" "${VAE_ARG[@]}" "$@"

echo "End: $(date)"
