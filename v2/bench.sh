#!/bin/bash
#SBATCH --job-name=bench
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
# Cost table (params, GFLOPs, train it/s, peak memory, sampling speed) for baseline vs method,
# all measured on the same GPU. Takes about 10-20 minutes. Submit from the folder with bench_cost.py:
#
#   sbatch --nodelist=gpu03 bench.sh                     # same GPU type as the S/2 training runs
#   sbatch --nodelist=gpu02 bench.sh                     # the node the B/2 baseline runs on
#   sbatch --exclude=gpu04 bench.sh --models SiT-XL/2 --batch-size 32
#
# Result: logs/bench_<jobid>.out (table) and results/bench/<node>_<jobid>.json
set -eo pipefail
if [ -z "${ROOT:-}" ]; then
    for d in "$PWD" "${SLURM_SUBMIT_DIR:-}"; do
        if [ -n "$d" ] && [ -f "$d/bench_cost.py" ]; then ROOT="$d"; break; fi
    done
fi
[ -n "${ROOT:-}" ] || { echo "ERROR: bench_cost.py not found; cd into the code folder first."; exit 1; }

source /media/ltnghia33/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sit}"
cd "$ROOT"

# same libcuda workaround as run.sh, so torch.compile works and the timing matches training
LIBCUDA=$( { /sbin/ldconfig -p 2>/dev/null || ldconfig -p 2>/dev/null; } | awk '/libcuda\.so\.1 /{print $NF; exit}')
if [ -n "$LIBCUDA" ] && [ -z "${TRITON_LIBCUDA_PATH:-}" ]; then
    LIBCUDA_DIR="${TMPDIR:-/tmp}/triton_libcuda_${SLURM_JOB_ID:-$$}"
    mkdir -p "$LIBCUDA_DIR" && ln -sf "$LIBCUDA" "$LIBCUDA_DIR/libcuda.so"
    export TRITON_LIBCUDA_PATH="$LIBCUDA_DIR"
fi

echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $(date)"
python bench_cost.py --out "results/bench/$(hostname)_${SLURM_JOB_ID:-local}.json" "$@"
echo "End $(date)"
