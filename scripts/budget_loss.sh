#!/bin/bash
#SBATCH --job-name=budget_loss
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
# Step 1c: loss per (noise level, token budget), budget-to-budget differences, alpha(t) and the
# extrapolating guidance scale, for one or more checkpoints (analyze_budgets.py). No sampling, no FID.
# Submit from the repository root:
#
#   sbatch scripts/budget_loss.sh --ckpt results/nested_uniform/checkpoints/0400000.pt \
#       results/nested_time/checkpoints/0400000.pt results/method/checkpoints/0400000.pt \
#       results/baseline/checkpoints/0400000.pt
#
# Result: table in logs/budget_loss_<jobid>.out, JSON per checkpoint in results/<exp>/budget_loss/,
# results/figures/budget_loss.csv / .png / .pdf
set -eo pipefail

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"

activate_env
cd "$SIT_ROOT"

echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | $(date)"
python -u analyze_budgets.py --data-path "$SIT_DATA" "$@"
echo "End $(date)"
