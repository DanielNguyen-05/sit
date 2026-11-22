#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=batch
#SBATCH --output=logs/bench/%x_%j.out
#SBATCH --error=logs/bench/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#
# Cost table (params, GFLOPs, train it/s, peak memory, sampling speed) for baseline vs method,
# all measured on the same GPU. Takes about 10-20 minutes. Submit from the repository root:
#
#   sbatch --nodelist=gpu03 scripts/bench.sh                     # same GPU type as the S/2 training runs
#   sbatch --nodelist=gpu02 scripts/bench.sh                     # the node the B/2 baseline runs on
#   sbatch --exclude=gpu04 scripts/bench.sh --models SiT-XL/2 --batch-size 32
#
# Result: logs/bench/bench_<jobid>.out (table) and results/bench/<node>_<jobid>.json
set -eo pipefail

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"

activate_env
cd "$SIT_ROOT"
setup_libcuda     # same as training, so torch.compile works and the timing matches

echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $(date)"
python bench_cost.py --out "$SIT_RESULTS/bench/$(hostname)_${SLURM_JOB_ID:-local}.json" "$@"
echo "End $(date)"
