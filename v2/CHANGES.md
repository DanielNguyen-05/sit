# v2: what changed from the files in the parent folder

Unchanged copies: `dataset.py`, `repack_latents.py`, `transport/`.

## models.py
- **Residual fix** in the V1 compressor and the decompressor: the gated FFN is now added to the
  attention stream (`q + norm2(h) * gate`) instead of replacing it. `legacy_residual=True`
  (`--legacy-residual`) reproduces the old forward exactly, for old checkpoints and for an
  old-vs-fixed ablation.
- **Baseline in the same codebase**: `compress=False, detailer=False`
  (`--no-compress --no-detailer`) is a plain SiT trunk; the extra modules are not even built
  (SiT-S/2: 32.96M parameters vs 40.45M).
- `need_weights=False` on the V1/decompressor attention calls (fast attention path).
- Classifier-free guidance is applied to all 4 latent channels (was the first 3).
- `RMSNorm` computes in fp32 so fp16 autocast cannot overflow inside the norm. Same parameters.
- `zero_init_final=True` (`--zero-init-final`) restores upstream SiT's zero-initialised output
  layer. Off by default, as before.

## train.py
- **Resume**: `--exp-name NAME` gives the run a fixed folder; if
  `results/NAME/checkpoints/latest.pt` exists the run continues from it (weights, EMA,
  optimizer, step count). Architecture flags that disagree with the checkpoint are refused.
- **Checkpoints**: `latest.pt` (full, overwritten every `--ckpt-every`) plus EMA-only
  milestones every `--keep-every` (default 50k). SiT-S/2: about 650MB + 160MB per milestone.
- **Precision**: `--amp auto|bf16|fp16|none`. `auto` uses fp16 with a GradScaler on V100 and
  bf16 on Ampere and newer.
- `--max-steps`, `--grad-clip`, `--no-compile`.
- Updates with non-finite gradients are skipped (never reach the weights or the EMA); 20 in a
  row stops the run.
- wandb, `ENTITY`/`PROJECT` and the VAE are only needed when actually used.

## sample.py, sample_ddp.py
- The architecture is rebuilt from the args stored in the checkpoint. Checkpoints written by the
  old `train.py` are treated as `legacy_residual=True`.

## run.sh
- Slurm script: `sbatch run.sh baseline|method|legacy [train.py flags]`. Usage is at the top of
  the file.

## Environment
- `requirements.txt` pins the versions this was tested with. **Use `diffusers==0.35.2`**: the
  current release (0.41) cannot be imported with torch 2.5.1, and `train.py` imports diffusers.

## Data preparation (only if latents.npy / labels.npy are missing)
- `encode_latents.py`, `encode.sh`, `download_shards.py`: build the two files from the Hugging
  Face ImageNet-1k parquet shards, one shard at a time.

## Not tested here
Tested on CPU only, with fake latents, under torch 2.5.1 / timm 1.0.30 / diffusers 0.35.2 /
Python 3.12: train, kill and resume, 2-process DDP for every variant (baseline, V1, V1 + aux,
V2, V3), fp16 and bf16 autocast (CPU autocast), `torch.compile` (CPU backend), checkpoint loading
in the samplers, and `run.sh` end to end with a simulated time-limit signal.

Not run: real GPUs (CUDA autocast, GradScaler on real fp16 gradients, throughput), real Slurm,
and the Hugging Face download in `encode_latents.py`.
