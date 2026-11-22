#!/bin/bash
#SBATCH --job-name=fid_batch
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
# Many FID evaluations as ONE Slurm job on ONE GPU (the cluster allows one GPU and at most 8 CPUs
# per job, and cancels many simultaneous jobs as spam): the configurations in a list file run one
# after the other. From the repository root:
#
#   sbatch --exclude=gpu04 scripts/fid_batch.sh configs/guidance_round3.txt
#   sbatch --exclude=gpu04 --dependency=afterany:<jobid> scripts/fid_batch.sh configs/...   # start after another job
#
# (If a job is ever given several GPUs, the list is split over them automatically.)
#
# List file: one configuration per line,  label | checkpoint | eval_fid.py flags   ("#" = comment).
# Resumable: a configuration whose result JSON already exists is skipped, so after a cancel or a
# timeout, submit the same command again. Per-configuration logs: logs/<job name>_<job id>_gpu<i>.log;
# the summary (one FID line per configuration) is in logs/<job name>_<job id>.out.
set -o pipefail
LIST=${1:-}
[ -n "$LIST" ] || { echo "usage: sbatch scripts/fid_batch.sh <list file>"; exit 2; }

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"
cd "$SIT_ROOT"
[ -f "$LIST" ] || { echo "ERROR: list file not found: $LIST"; exit 1; }
require_fid_ref
activate_env
export HF_HUB_OFFLINE=1

# configurations: drop comments and blank lines, check every checkpoint before any GPU time is spent
CONFIGS=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; [ -z "${line//[[:space:]]/}" ] && continue
    IFS='|' read -r label ckpt flags <<< "$line"
    label=$(echo "$label" | xargs); ckpt=$(echo "$ckpt" | xargs); flags=$(echo "$flags" | xargs)
    [ -f "$ckpt" ] || { echo "ERROR: checkpoint not found for '$label': $ckpt"; exit 1; }
    CONFIGS+=("$label|$ckpt|$flags")
done < "$LIST"
[ ${#CONFIGS[@]} -gt 0 ] || { echo "ERROR: no configuration in $LIST"; exit 1; }

# GPUs of this job (Slurm sets CUDA_VISIBLE_DEVICES to the allocated devices)
IFS=',' read -ra DEVS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NG=${#DEVS[@]}
TAG="${SLURM_JOB_NAME:-fid_batch}_${SLURM_JOB_ID:-local}"
echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $NG GPU(s): ${DEVS[*]} | ${#CONFIGS[@]} configurations from $LIST | $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null

lane() {   # lane <lane index>: runs configurations lane, lane+NG, lane+2NG, ... on one GPU
    local i=$1 dev=${DEVS[$1]} log="logs/${TAG}_gpu$1.log" k label ckpt flags
    for ((k = i; k < ${#CONFIGS[@]}; k += NG)); do
        IFS='|' read -r label ckpt flags <<< "${CONFIGS[$k]}"
        echo "######## [gpu $dev] $label  $(date)" >> "$log"
        # plain python (not torchrun): one process per GPU, no port clashes between lanes
        if CUDA_VISIBLE_DEVICES=$dev python -u eval_fid.py --ckpt "$ckpt" --ref-stats "$REF_STATS" --vae "$VAE_DIR" \
                $flags >> "$log" 2>&1; then
            echo "[$label] $(grep -E '^FID|^skip' "$log" | tail -1)"
        else
            echo "[$label] FAILED (see $log)"
        fi
    done
}

for ((i = 0; i < NG; i++)); do lane $i & done
wait
echo "End $(date)"
grep -q "FAILED" "logs/${TAG}.out" 2>/dev/null && exit 1 || exit 0
