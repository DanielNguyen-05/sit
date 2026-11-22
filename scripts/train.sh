#!/bin/bash
#SBATCH --job-name=sit
#SBATCH --partition=batch
#SBATCH --output=logs/train/%x_%j.out
#SBATCH --error=logs/train/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300
#
# Train SiT on the Slurm cluster. Always submit from the repository root (folder with train.py):
#
#   cd /media/ltnghia33/sit
#   sbatch --job-name=base   scripts/train.sh baseline          # plain SiT, same codebase
#   sbatch --job-name=method scripts/train.sh method            # compressor (residual fixed)
#   sbatch --job-name=legacy scripts/train.sh legacy            # compressor as in the old code
#   EXP=method_aux sbatch scripts/train.sh method --aux-recon   # anything after the mode goes to train.py
#   MODEL=SiT-B/2 EXP=method_B2 sbatch scripts/train.sh method
#   sbatch --gres=gpu:4 --cpus-per-task=32 --mem=128G scripts/train.sh method   # 4 GPUs, same global batch
#   EXP=nested_uniform sbatch scripts/train.sh method --query-budgets 8,16,32,64 --budget-mode uniform --ckpt-every 25000
#
# Short test inside an interactive GPU session (200 steps, separate smoke_* folder):
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash
#   SMOKE=1 bash scripts/train.sh method
#
# Print the command without running it (works on the login node):
#   DRY_RUN=1 bash scripts/train.sh baseline
#
# Resubmitting the same command continues from results/<EXP>/checkpoints/latest.pt.
# Settings (environment variables): EXP MODEL STEPS BATCH AMP COMPILE WANDB, plus the paths in env.sh.

set -eo pipefail

MODE=${1:-}
case "$MODE" in
    baseline) MODE_FLAGS="--no-compress --no-detailer" ;;
    method)   MODE_FLAGS="" ;;
    legacy)   MODE_FLAGS="--legacy-residual" ;;
    *) echo "usage: sbatch scripts/train.sh {baseline|method|legacy} [extra train.py flags]"; exit 2 ;;
esac
shift

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"
cd "$SIT_ROOT"

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
     --data-path "$SIT_DATA" --results-dir "$SIT_RESULTS" --exp-name "$EXP"
     --model "$MODEL" --global-batch-size "$BATCH" --max-steps "$STEPS"
     --num-workers "$WORKERS" --amp "$AMP"
     --log-every "$LOG_EVERY" --ckpt-every "$CKPT_EVERY" --keep-every "$KEEP_EVERY"
     --sample-every 0
     $MODE_FLAGS $EXTRA_FLAGS $SMOKE_FLAGS "$@")

echo "================================"
echo "Job ID     : ${SLURM_JOB_ID:-LOCAL}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Repository : $SIT_ROOT"
echo "Data       : $SIT_DATA"
echo "Run folder : $SIT_RESULTS/$EXP"
echo "GPUs       : $NGPU   CPUs: $CPUS   workers/GPU: $WORKERS"
echo "Command    : ${CMD[*]}"
echo "================================"

if [ "${DRY_RUN:-0}" = "1" ]; then exit 0; fi

if [ ! -f "$SIT_DATA/latents.npy" ] || [ ! -f "$SIT_DATA/labels.npy" ]; then
    echo "ERROR: latents.npy / labels.npy not found in $SIT_DATA (set DATA=/path/to/folder)"; exit 1
fi

# Slurm sends USR1 five minutes before the time limit (see --signal above). The run is then
# resubmitted and continues from latest.pt; at most CKPT_EVERY steps are repeated.
# RESUB caps the chain so a run that makes no progress cannot resubmit itself forever.
PID=""
resubmit() {
    echo "Time limit close at $(date)."
    if [ -n "${SLURM_JOB_ID:-}" ] && [ "${RESUB:-0}" -lt 5 ]; then
        echo "Resubmitting to continue from latest.pt (resubmission $(( ${RESUB:-0} + 1 )) of at most 5)."
        # keep the node choice of this job (e.g. --exclude=gpu04), so the run stays on one GPU type
        EXC=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'ExcNodeList=[^ ]*' | cut -d= -f2)
        [ "$EXC" = "(null)" ] && EXC=""
        (cd "$SIT_ROOT" && RESUB=$(( ${RESUB:-0} + 1 )) EXP="$EXP" MODEL="$MODEL" STEPS="$STEPS" BATCH="$BATCH" AMP="$AMP" \
            sbatch --job-name="${SLURM_JOB_NAME:-sit}" \
            --gres="gpu:$NGPU" --cpus-per-task="$CPUS" ${SLURM_MEM_PER_NODE:+--mem="${SLURM_MEM_PER_NODE}M"} \
            ${EXC:+--exclude="$EXC"} scripts/train.sh "$MODE" "$@")
    else
        echo "Not resubmitting (limit of 5 reached). Submit the same command again to continue."
    fi
    [ -n "$PID" ] && kill -TERM "$PID" 2>/dev/null
    wait "$PID" 2>/dev/null || true
    exit 0
}
trap 'resubmit "$@"' USR1

activate_env
setup_libcuda

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
