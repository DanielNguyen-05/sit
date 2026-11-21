#!/bin/bash
# Submits the self-guidance experiments (FID-10K, one Slurm job each) on the existing 400K checkpoints.
# Run on the login node from the repository root:   bash scripts/submit_guidance_sweep.sh
# DRY_RUN=1 bash scripts/submit_guidance_sweep.sh    prints the sbatch commands only.
#
# A  scale sweep            weak K=16, s = 1.25 / 2 / 3            (s = 1.5 already done: FID 49.63)
# B  weak budget            weak K = 8 / 32 at s = 1.5
# C  guidance interval      guide only for t in [0.3, 0.9], where budgets differ (analyze_budgets.py)
# D  cheap CFG (ELIT)       weak branch without class label
# E  references             CFG on the baseline and on the nested model at K=64 (2 forwards per step)
# F  nested_time            fixed K = 16 / 32 / 64 (jobs 84483-84485 were cancelled)
# G  step 1c again          analyze_budgets.py on all four checkpoints (combined figure + method/baseline)
set -eo pipefail
[ -f train.py ] && [ -f scripts/fid.sh ] || { echo "run from the repository root"; exit 1; }

NU=results/nested_uniform/checkpoints/0400000.pt
NT=results/nested_time/checkpoints/0400000.pt
BL=results/baseline/checkpoints/0400000.pt
ME=results/method/checkpoints/0400000.pt
SB=(sbatch --exclude="${EXCLUDE:-gpu04}")        # gpu04 is a V100: about 3-4x slower
N="--num-samples 10000"

submit() {   # submit <job-name> <FID_ARGS> <checkpoint>
    if [ "${DRY_RUN:-0}" = 1 ]; then echo "FID_ARGS=\"$2\" ${SB[*]} --job-name=$1 scripts/fid.sh $3"
    else FID_ARGS="$2" "${SB[@]}" --job-name="$1" scripts/fid.sh "$3"; fi
}

for s in 1.25 2.0 3.0; do submit sg_A "$N --num-queries 64 --weak-queries 16 --guidance-scale $s" $NU; done
for k in 8 32;          do submit sg_B "$N --num-queries 64 --weak-queries $k --guidance-scale 1.5" $NU; done
for s in 1.5 2.0;       do submit sg_C "$N --num-queries 64 --weak-queries 16 --guidance-scale $s --guidance-interval 0.3,0.9" $NU; done
submit sg_D "$N --num-queries 64 --weak-queries 16 --guidance-scale 1.5 --weak-uncond" $NU
for s in 1.25 1.5;      do submit sg_E "$N --cfg-scale $s" $BL; done
submit sg_E "$N --num-queries 64 --cfg-scale 1.5" $NU
for q in 16 32 64;      do submit fid_nt "$N --num-queries $q" $NT; done

if [ "${DRY_RUN:-0}" = 1 ]; then echo "${SB[*]} scripts/budget_loss.sh --ckpt $NU $NT $ME $BL"
else "${SB[@]}" scripts/budget_loss.sh --ckpt $NU $NT $ME $BL; fi
