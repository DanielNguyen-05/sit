# Sourced by every script in scripts/. The ONLY file with machine-specific paths.
#
# All scripts are started from the repository root (the folder with train.py):
#
#   cd /media/ltnghia33/sit
#   sbatch scripts/train.sh method
#
# Anything below can be overridden for one command by exporting it first, e.g.
#   DATA=/scratch/me/imagenet_latents sbatch scripts/train.sh baseline

# ---------------------------------------------------------------- repository root
# Under sbatch, BASH_SOURCE points to Slurm's private copy of the script, so the root is found
# from the submit folder first, then the current folder, then the location of this file.
if [ -z "${SIT_ROOT:-}" ]; then
    for _d in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
        if [ -n "$_d" ] && [ -f "$_d/train.py" ] && [ -f "$_d/scripts/env.sh" ]; then SIT_ROOT="$_d"; break; fi
    done
    unset _d
fi
if [ -z "${SIT_ROOT:-}" ]; then
    echo "ERROR: repository root not found. cd into the folder with train.py and submit from there." >&2
    exit 1
fi
export SIT_ROOT

# ---------------------------------------------------------------- environment
CONDA_SH=${CONDA_SH:-/media/ltnghia33/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-sit}

# ---------------------------------------------------------------- data and outputs
# The Python side reads the same variables (paths.py), so both always agree.
export SIT_DATA=${DATA:-${SIT_DATA:-$SIT_ROOT/data}}           # latents.npy + labels.npy
export SIT_FID_REF=${FID_REF:-${SIT_FID_REF:-$SIT_ROOT/fid_ref}}  # FID stats, Inception, SD-VAE
export SIT_RESULTS=${RESULTS:-${SIT_RESULTS:-$SIT_ROOT/results}}  # results/<exp>/...
export SIT_LOGS=${SIT_LOGS:-$SIT_ROOT/logs}                       # Slurm logs (see #SBATCH --output)
PARQUET=${PARQUET:-/media/ltnghia33/imagenet_parquet}             # only for encode.sh / tools/download_shards.py

REF_STATS="$SIT_FID_REF/VIRTUAL_imagenet256_labeled_stats.npz"
VAE_DIR="$SIT_FID_REF/sd-vae-ft-ema"
export TORCH_HOME="$SIT_FID_REF/torch"     # Inception weights for FID (~/.cache on the server gets cleaned)

mkdir -p "$SIT_LOGS"

# ---------------------------------------------------------------- helpers
activate_env() {
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    else
        echo "WARNING: $CONDA_SH not found; using the python on PATH ($(command -v python || echo none))." >&2
    fi
}

# torch.compile (Triton) links against "libcuda.so", but the GPU nodes only ship "libcuda.so.1".
# Give Triton a private folder with the missing name (local /tmp, so jobs never share it).
setup_libcuda() {
    local lib dir
    lib=$( { /sbin/ldconfig -p 2>/dev/null || ldconfig -p 2>/dev/null; } | awk '/libcuda\.so\.1 /{print $NF; exit}')
    if [ -n "$lib" ] && [ -z "${TRITON_LIBCUDA_PATH:-}" ]; then
        dir="${TMPDIR:-/tmp}/triton_libcuda_${SLURM_JOB_ID:-$$}"
        mkdir -p "$dir" && ln -sf "$lib" "$dir/libcuda.so"
        export TRITON_LIBCUDA_PATH="$dir"
        echo "TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH -> $lib"
    fi
}

# Fails with a clear message instead of a Python traceback halfway through a job.
require_fid_ref() {
    local ok=1
    [ -f "$REF_STATS" ] || { echo "ERROR: $REF_STATS missing." >&2; ok=0; }
    [ -f "$VAE_DIR/config.json" ] && [ -f "$VAE_DIR/diffusion_pytorch_model.safetensors" ] || {
        echo "ERROR: SD-VAE files missing in $VAE_DIR." >&2; ok=0; }
    [ -f "$TORCH_HOME/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth" ] || {
        echo "ERROR: Inception weights missing in $TORCH_HOME." >&2; ok=0; }
    [ "$ok" = 1 ] || { echo "       Run 'bash scripts/prepare_fid.sh' once on the login node." >&2; exit 1; }
}
