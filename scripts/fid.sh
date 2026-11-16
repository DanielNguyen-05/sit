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
# (results/<exp>/fid/). Run scripts/prepare_fid.sh once first. Submit from the repository root:
#
#   sbatch --exclude=gpu04 scripts/fid.sh results/baseline/checkpoints/0400000.pt results/method/checkpoints/0400000.pt
#   sbatch --exclude=gpu04 scripts/fid.sh results/method/checkpoints/0*.pt          # every milestone of a run
#   SMOKE=1 sbatch scripts/fid.sh results/baseline/checkpoints/0400000.pt   # 1,000 samples, quick check
#
# Token-budget evaluation (results go to results/<exp>/fid_budget/), 10k samples:
#   FID_ARGS="--num-samples 10000 --num-queries 32" sbatch scripts/fid.sh results/nested_time/checkpoints/0400000.pt
#   FID_ARGS="--num-samples 10000 --num-queries 64 --weak-queries 16 --guidance-scale 1.5" sbatch scripts/fid.sh ...
#
# Roughly 1-1.5 h per S/2 checkpoint on one A100 (250 sampling steps x 50k images, fp32/tf32);
# FID_ARGS="--amp bf16" about halves that. Already-scored checkpoints are skipped.
# Other eval_fid.py flags: FID_ARGS="--mode ODE --sampling-method heun2 --num-sampling-steps 50" sbatch scripts/fid.sh ...
set -eo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch scripts/fid.sh <checkpoint.pt> [more.pt ...]"; exit 2; }

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"

# Checkpoint paths may be relative to the submit folder; make them absolute before cd.
CKPTS=()
for c in "$@"; do
    [ -f "$c" ] || { echo "ERROR: checkpoint not found: $c"; exit 1; }
    CKPTS+=("$(cd "$(dirname "$c")" && pwd)/$(basename "$c")")
done

require_fid_ref
activate_env
cd "$SIT_ROOT"
export HF_HUB_OFFLINE=1          # VAE is loaded by folder path, Inception from TORCH_HOME: no network needed
NGPU=${SLURM_GPUS_ON_NODE:-1}

echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | $(date)"
if [ "${SMOKE:-0}" = "1" ]; then
    for c in "${CKPTS[@]}"; do
        torchrun --standalone --nproc_per_node="$NGPU" eval_fid.py --ckpt "$c" --ref-stats "$REF_STATS" --vae "$VAE_DIR" \
            --num-samples 1000 --out "$(dirname "$(dirname "$c")")/fid/smoke_$(basename "$c" .pt).json" --overwrite ${FID_ARGS:-}
    done
else
    torchrun --standalone --nproc_per_node="$NGPU" eval_fid.py --ckpt "${CKPTS[@]}" --ref-stats "$REF_STATS" --vae "$VAE_DIR" ${FID_ARGS:-}
fi
echo "End $(date)"
