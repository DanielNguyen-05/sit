"""
Every default folder of the project, in one place, relative to this repository.

    <repo>/data/            latents.npy + labels.npy (training data)
    <repo>/fid_ref/         FID reference statistics, Inception weights, SD-VAE decoder
    <repo>/results/<exp>/   checkpoints/, fid/, fid_budget/, log.txt of each run
    <repo>/results/figures/ paper figures (tools/plot_*.py)
    <repo>/logs/            Slurm stdout / stderr
    <repo>/pretrained_models/  released SiT weights (only for sample.py without --ckpt)

Each one can be moved elsewhere with an environment variable (SIT_DATA, SIT_FID_REF,
SIT_RESULTS, SIT_LOGS, SIT_PRETRAINED) or with the matching command-line flag of each script,
so nothing depends on the folder the command is started from.
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("SIT_DATA", os.path.join(REPO, "data"))
FID_REF_DIR = os.environ.get("SIT_FID_REF", os.path.join(REPO, "fid_ref"))
RESULTS_DIR = os.environ.get("SIT_RESULTS", os.path.join(REPO, "results"))
LOGS_DIR = os.environ.get("SIT_LOGS", os.path.join(REPO, "logs"))
PRETRAINED_DIR = os.environ.get("SIT_PRETRAINED", os.path.join(REPO, "pretrained_models"))

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
REF_STATS = os.path.join(FID_REF_DIR, "VIRTUAL_imagenet256_labeled_stats.npz")
VAE_DIR = os.path.join(FID_REF_DIR, "sd-vae-ft-ema")


def vae_source(name="ema"):
    """Local SD-VAE folder when it exists (no network needed), else the Hugging Face id."""
    if os.path.isdir(name):
        return name
    if name == "ema" and os.path.isfile(os.path.join(VAE_DIR, "config.json")):
        return VAE_DIR
    return f"stabilityai/sd-vae-ft-{name}"
