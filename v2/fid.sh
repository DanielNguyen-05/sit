#!/bin/bash
#SBATCH --job-name=fid
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#
# FID-50K for one or more checkpoints, nothing written to disk except one small JSON per checkpoint
# (results/<exp>/fid/). Run prepare_fid.sh once first. Submit from the folder with eval_fid.py:
#
#   sbatch --exclude=gpu04 fid.sh results/baseline/checkpoints/0400000.pt results/method/checkpoints/0400000.pt
#   sbatch --exclude=gpu04 fid.sh results/method/checkpoints/0*.pt          # every milestone of a run
#   SMOKE=1 sbatch fid.sh results/baseline/checkpoints/0400000.pt   # 1,000 samples, quick check
#
# Roughly 1-1.5 h per S/2 checkpoint on one A100 (250 sampling steps x 50k images, fp32/tf32);
# FID_ARGS="--amp bf16" about halves that. Already-scored checkpoints are skipped.
# Extra eval_fid.py flags: FID_ARGS="--mode ODE --sampling-method heun2 --num-sampling-steps 50" sbatch fid.sh ...
set -eo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch fid.sh <checkpoint.pt> [more.pt ...]"; exit 2; }

if [ -z "${ROOT:-}" ]; then
    for d in "$PWD" "${SLURM_SUBMIT_DIR:-}"; do
        if [ -n "$d" ] && [ -f "$d/eval_fid.py" ]; then ROOT="$d"; break; fi
    done
fi
[ -n "${ROOT:-}" ] || { echo "ERROR: eval_fid.py not found; cd into the code folder first."; exit 1; }
REF=${REF:-/media/ltnghia33/sit/fid_ref/VIRTUAL_imagenet256_labeled_stats.npz}
[ -f "$REF" ] || { echo "ERROR: $REF missing; run prepare_fid.sh on the login node first."; exit 1; }
VAE_FILE=$(dirname "$REF")/vae_path.txt
[ -f "$VAE_FILE" ] || { echo "ERROR: $VAE_FILE missing; rerun prepare_fid.sh on the login node (it is quick now)."; exit 1; }
VAE_DIR=$(cat "$VAE_FILE")
[ -f "$VAE_DIR/config.json" ] || { echo "ERROR: VAE folder $VAE_DIR has no config.json; rerun prepare_fid.sh."; exit 1; }

source /media/ltnghia33/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sit}"
cd "$ROOT"
export HF_HUB_OFFLINE=1          # VAE is loaded by folder path, Inception from the torch cache: no network needed
NGPU=${SLURM_GPUS_ON_NODE:-1}



echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | $(date)"
if [ "${SMOKE:-0}" = "1" ]; then
    for c in "$@"; do
        torchrun --standalone --nproc_per_node="$NGPU" eval_fid.py --ckpt "$c" --ref-stats "$REF" --vae "$VAE_DIR" \
            --num-samples 1000 --out "$(dirname "$(dirname "$c")")/fid/smoke_$(basename "$c" .pt).json" --overwrite ${FID_ARGS:-}
    done
else
    torchrun --standalone --nproc_per_node="$NGPU" eval_fid.py --ckpt "$@" --ref-stats "$REF" --vae "$VAE_DIR" ${FID_ARGS:-}
fi
echo "End $(date)"
