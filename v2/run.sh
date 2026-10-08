#!/bin/bash
#SBATCH --job-name=sit
#SBATCH --partition=batch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#
# Train SiT on the Slurm cluster. Always submit from the folder that contains train.py,
# and create logs/ first (Slurm does not create it):
#
#   cd /media/ltnghia33/sit/v2 && mkdir -p logs
#
#   sbatch --job-name=base   run.sh baseline          # plain SiT, same codebase
#   sbatch --job-name=method run.sh method            # compressor (residual fixed)
#   sbatch --job-name=legacy run.sh legacy            # compressor as in the old code
#   EXP=method_aux sbatch run.sh method --aux-recon   # anything after the mode goes to train.py
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=128G run.sh method   # 4 GPUs, same global batch
#
# Short test inside an interactive GPU session (200 steps, separate smoke_* folder):
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash
#   SMOKE=1 bash run.sh method
#
# Print the command without running it (works on the login node):
#   DRY_RUN=1 bash run.sh baseline
#
# Resubmitting the same command continues from results/<EXP>/checkpoints/latest.pt.
# Settings (environment variables): EXP MODEL STEPS BATCH AMP COMPILE DATA RESULTS WANDB CONDA_ENV

set -eo pipefail

MODE=${1:-}
case "$MODE" in
    baseline) MODE_FLAGS="--no-compress --no-detailer" ;;
    method)   MODE_FLAGS="" ;;
    legacy)   MODE_FLAGS="--legacy-residual" ;;
    *) echo "usage: run.sh {baseline|method|legacy} [extra train.py flags]"; exit 2 ;;
esac
shift

# Folder with train.py: the current folder (sbatch starts in the submit folder), else the
# submit folder, else wherever this script lives.
if [ -z "${ROOT:-}" ]; then
    for d in "$PWD" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
        if [ -n "$d" ] && [ -f "$d/train.py" ]; then ROOT="$d"; break; fi
    done
fi
if [ -z "${ROOT:-}" ]; then echo "ERROR: train.py not found; cd into the code folder first."; exit 1; fi
DATA=${DATA:-/media/ltnghia33/sit}               # folder with latents.npy + labels.npy
RESULTS=${RESULTS:-$ROOT/results}
CONDA_ENV=${CONDA_ENV:-sit}
MODEL=${MODEL:-SiT-S/2}
STEPS=${STEPS:-400000}
BATCH=${BATCH:-256}
AMP=${AMP:-auto}                                 # auto = fp16 on V100, bf16 on Ampere and newer
EXP=${EXP:-$MODE}                                # run folder name; keep it fixed to be able to resume
CKPT_EVERY=5000
KEEP_EVERY=50000
LOG_EVERY=100

SMOKE_FLAGS=""
if [ "${SMOKE:-0}" = "1" ]; then
    # always a fresh 200-step run, so repeating the test never just reports "already finished"
    EXP="smoke_$EXP"; STEPS=200; CKPT_EVERY=100; KEEP_EVERY=0; LOG_EVERY=20; SMOKE_FLAGS="--resume none"
fi

NGPU=${SLURM_GPUS_ON_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}
WORKERS=$(( CPUS / NGPU - 1 ))
[ "$WORKERS" -lt 2 ] && WORKERS=2
[ "$WORKERS" -gt 8 ] && WORKERS=8

EXTRA_FLAGS=""
[ "${COMPILE:-1}" = "0" ] && EXTRA_FLAGS="$EXTRA_FLAGS --no-compile"
[ "${WANDB:-0}" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --wandb"     # needs ENTITY, PROJECT, WANDB_KEY

CMD=(torchrun --standalone --nproc_per_node="$NGPU" train.py
     --data-path "$DATA" --results-dir "$RESULTS" --exp-name "$EXP"
     --model "$MODEL" --global-batch-size "$BATCH" --max-steps "$STEPS"
     --num-workers "$WORKERS" --amp "$AMP"
     --log-every "$LOG_EVERY" --ckpt-every "$CKPT_EVERY" --keep-every "$KEEP_EVERY"
     --sample-every 0
     $MODE_FLAGS $EXTRA_FLAGS $SMOKE_FLAGS "$@")

echo "================================"
echo "Job ID     : ${SLURM_JOB_ID:-LOCAL}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Run folder : $RESULTS/$EXP"
echo "GPUs       : $NGPU   CPUs: $CPUS   workers/GPU: $WORKERS"
echo "Command    : ${CMD[*]}"
echo "================================"

if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi

if [ ! -f "$DATA/latents.npy" ] || [ ! -f "$DATA/labels.npy" ]; then
    echo "ERROR: latents.npy / labels.npy not found in $DATA (set DATA=/path/to/folder)"; exit 1
fi

# Slurm sends USR1 five minutes before the time limit (see --signal above). The run is then
# resubmitted and continues from latest.pt; at most CKPT_EVERY steps are repeated.
# RESUB caps the chain so a run that makes no progress cannot resubmit itself forever.
PID=""
resubmit() {
    echo "Time limit close at $(date)."
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "${RESUB:-0}" -lt 5 ]; then
        echo "Resubmitting to continue from latest.pt (resubmission $(( ${RESUB:-0} + 1 )) of at most 5)."
        RESUB=$(( ${RESUB:-0} + 1 )) EXP="$EXP" sbatch --job-name="${SLURM_JOB_NAME:-sit}" \
            --gres="gpu:$NGPU" --cpus-per-task="$CPUS" ${SLURM_MEM_PER_NODE:+--mem="${SLURM_MEM_PER_NODE}M"} \
            "$ROOT/run.sh" "$MODE" "$@"
    else
        echo "Not resubmitting (limit of 5 reached). Submit the same command again to continue."
    fi
    [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null || true
    exit 0
}
trap 'resubmit "$@"' USR1

source /media/ltnghia33/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT"
mkdir -p logs "$RESULTS"

which python
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi --query-gpu=index,name,memory.total --format=csv; fi

"${CMD[@]}" &
PID=$!
set +e
wait "$PID"
RC=$?
set -e

echo "================================"
echo "End time : $(date)   exit code: $RC"
echo "================================"
exit $RC
