#!/bin/bash
#SBATCH --job-name=guid_sweep
#SBATCH --partition=batch
#SBATCH --output=logs/sweep/%x_%j.out
#SBATCH --error=logs/sweep/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#
# Self-guidance experiments as ONE Slurm job on ONE GPU: every configuration runs one after the
# other (never many jobs at once, which the cluster treats as spam). Submit from the repository root:
#
#   sbatch --exclude=gpu04 scripts/guidance_sweep.sh
#
# Resumable: a configuration whose result JSON already exists is skipped (eval_fid.py does this),
# so if the job is cancelled or times out, submit the same command again and it continues.
# About 7-9 h on one A100 for everything below. Comment out lines to run fewer.
#
# Order = priority (the most important comparisons first):
#   G  analyze_budgets.py on all four checkpoints (minutes)
#   E  CFG on the baseline (1.25, 1.5) and on the nested model at K=64      references at equal cost
#   A  self-guidance scale s = 1.25 / 2 / 3 (weak K=16)                      s = 1.5 already done
#   C  guide only for t in [0.3, 0.9] (s = 1.5, 2)                           the new idea
#   D  weak branch without class label (ELIT's cheap CFG)
#   B  weak K = 8 / 32
#   F  nested_time fixed K = 16 / 32 / 64
set -o pipefail

_r=${SIT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}; [ -f "$_r/scripts/env.sh" ] || _r="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_r/scripts/env.sh"
require_fid_ref
activate_env
cd "$SIT_ROOT"
export HF_HUB_OFFLINE=1

NU=results/nested_uniform/checkpoints/0400000.pt
NT=results/nested_time/checkpoints/0400000.pt
BL=results/baseline/checkpoints/0400000.pt
ME=results/method/checkpoints/0400000.pt
N="--num-samples 10000"
FAILED=()

fid() {   # fid <label> <checkpoint> <eval_fid.py flags...>
    local label=$1 ckpt=$2; shift 2
    echo; echo "######## $label  $(date)"
    torchrun --standalone --nproc_per_node=1 eval_fid.py --ckpt "$ckpt" --ref-stats "$REF_STATS" --vae "$VAE_DIR" $N "$@" \
        || { echo "!!! $label failed (exit $?), continuing"; FAILED+=("$label"); }
}

echo "Job ${SLURM_JOB_ID:-LOCAL} on $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) | $(date)"

# G: loss per (noise level, budget); skipped if the method/baseline results already exist
if [ -f results/method/budget_loss/0400000_n4096.json ] && [ -f results/baseline/budget_loss/0400000_n4096.json ]; then
    echo "######## G  already done, skipping"
else
    echo "######## G  analyze_budgets.py  $(date)"
    python -u analyze_budgets.py --data-path "$SIT_DATA" --ckpt $NU $NT $ME $BL || FAILED+=("G")
fi

fid "E  baseline CFG 1.25"     $BL --cfg-scale 1.25
fid "E  baseline CFG 1.5"      $BL --cfg-scale 1.5
fid "E  nested K=64 CFG 1.5"   $NU --num-queries 64 --cfg-scale 1.5

for s in 1.25 2.0 3.0; do
    fid "A  self-guidance s=$s"  $NU --num-queries 64 --weak-queries 16 --guidance-scale $s
done
for s in 1.5 2.0; do
    fid "C  interval 0.3-0.9 s=$s" $NU --num-queries 64 --weak-queries 16 --guidance-scale $s --guidance-interval 0.3,0.9
done
fid "D  weak without label"    $NU --num-queries 64 --weak-queries 16 --guidance-scale 1.5 --weak-uncond
for k in 8 32; do
    fid "B  weak K=$k"           $NU --num-queries 64 --weak-queries $k --guidance-scale 1.5
done
for q in 16 32 64; do
    fid "F  nested_time K=$q"    $NT --num-queries $q
done

echo; echo "End $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
