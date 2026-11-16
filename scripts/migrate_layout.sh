#!/bin/bash
# One-time move from the old layout (code in v2/, data at the root) to the current one.
# Run ONCE on every machine that has an old copy, after `git pull`, from the repository root:
#
#   cd /media/ltnghia33/sit
#   git pull
#   bash scripts/migrate_layout.sh            # shows what it would do
#   APPLY=1 bash scripts/migrate_layout.sh    # does it
#
#   latents.npy, labels.npy  -> data/
#   v2/results/<exp>/        -> results/<exp>/
#   v2/logs/*                -> logs/
#
# Only `mv` inside the same disk (instant, nothing is copied) and never overwrites: an item that
# already exists at the destination is reported and left where it is. Safe to run again.
# No training or FID job may be running while you do this (their paths would disappear).
set -eo pipefail

[ -f train.py ] && [ -f scripts/env.sh ] || { echo "ERROR: run from the repository root (folder with train.py)."; exit 1; }
APPLY=${APPLY:-0}
CONFLICTS=0

do_mv() {   # do_mv <src> <dst>
    local src=$1 dst=$2
    [ -e "$src" ] || return 0
    if [ -e "$dst" ]; then
        echo "  SKIP  $src  (already exists: $dst)"; CONFLICTS=$((CONFLICTS + 1)); return 0
    fi
    echo "  mv    $src -> $dst"
    if [ "$APPLY" = 1 ]; then mkdir -p "$(dirname "$dst")" && mv "$src" "$dst"; fi
}

if command -v squeue >/dev/null 2>&1 && [ -n "$(squeue -u "$USER" -h 2>/dev/null)" ]; then
    echo "WARNING: you have Slurm jobs queued or running:"; squeue -u "$USER"
    [ "$APPLY" = 1 ] && { echo "Stop them first (or wait), then rerun."; exit 1; }
fi

echo "== data"
do_mv latents.npy data/latents.npy
do_mv labels.npy  data/labels.npy

echo "== results"
if [ -d v2/results ]; then
    for d in v2/results/* v2/results/.[!.]*; do [ -e "$d" ] && do_mv "$d" "results/$(basename "$d")"; done
fi

echo "== logs"
if [ -d v2/logs ]; then
    for f in v2/logs/* v2/logs/.[!.]*; do [ -e "$f" ] && do_mv "$f" "logs/$(basename "$f")"; done
fi

if [ "$APPLY" = 1 ]; then
    rmdir v2/results v2/logs 2>/dev/null || true
    find v2 -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find v2 -name .DS_Store -delete 2>/dev/null || true
    find v2 -depth -type d -empty -delete 2>/dev/null || true
    if [ -d v2 ]; then
        echo "v2/ still has files git does not track (left alone, check them):"; find v2 -type f | head -20
    fi
    rm -f fid_ref/vae_path.txt    # no longer read: the VAE is always fid_ref/sd-vae-ft-ema
fi

echo
if [ "$APPLY" = 1 ]; then echo "Done ($CONFLICTS skipped)."; else echo "Dry run ($CONFLICTS conflicts). Run again with APPLY=1 to move."; fi
