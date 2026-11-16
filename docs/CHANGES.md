# What changed from upstream SiT

Written when this code lived in `v2/`; file names below are updated to the current layout
(`scripts/`, `tools/`, see the README). Paths: `paths.py` and `scripts/env.sh`.

## Layout refactor (Nov 2026)
- `v2/` became the repository root; the pre-fix code that sat next to it was removed (it is in git
  history before the `refactor-layout` branch).
- Data in `data/`, runs in `results/`, Slurm logs in `logs/`, FID files in `fid_ref/`, all relative to
  the repository. `scripts/migrate_layout.sh` moves an old copy.
- `run.sh` -> `scripts/train.sh`; all Slurm scripts are submitted from the repository root.
- `train.py --data-path`, `--results-dir`, `eval_fid.py --ref-stats` and the VAE now default to the
  repository folders; `repack_latents.py` takes `--src/--dst` instead of hard-coded paths.
- `fid_ref/vae_path.txt` is no longer used (the VAE is always `fid_ref/sd-vae-ft-ema`).

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

## scripts/train.sh (was run.sh)
- Slurm script: `sbatch scripts/train.sh baseline|method|legacy [train.py flags]`. Usage is at the top of
  the file.

## Environment
- `requirements.txt` pins the versions this was tested with. **Use `diffusers==0.35.2`**: the
  current release (0.41) cannot be imported with torch 2.5.1, and `train.py` imports diffusers.

## Data preparation (only if latents.npy / labels.npy are missing)
- `tools/encode_latents.py`, `scripts/encode.sh`, `tools/download_shards.py`: build the two files from the Hugging
  Face ImageNet-1k parquet shards, one shard at a time.

## FID
- `scripts/prepare_fid.sh` (login node, once): installs pytorch-fid, puts the Inception weights and the VAE
  in `fid_ref/` (not `~/.cache`, which the server cleans), and keeps only mu / sigma (~34MB) of
  OpenAI's ADM ImageNet-256 reference batch.
- `eval_fid.py` + `scripts/fid.sh`: FID-50K computed in memory (no images written). Defaults = the SiT
  paper's 400K setting: SDE Euler-Maruyama, 250 steps, no CFG, 50k samples. Same seeds for every
  checkpoint. One JSON per checkpoint in `results/<exp>/fid/`.
- The Fréchet distance is written out in `eval_fid.py` (pytorch-fid 0.3.0 calls a scipy function
  signature that current scipy removed). Checked against an eigenvalue computation (rel. diff 1e-15)
  and against itself (own statistics as reference gives 0.00).

## Paper tables and figures
- `bench_cost.py` + `scripts/bench.sh`: cost table, baseline vs method on one GPU: parameters, forward
  GFLOPs (DiT convention; baseline S/B/L/XL give 6.05 / 23.00 / 80.69 / 118.62, matching DiT),
  training it/s and peak memory for a real train step, sampling speed at 250 NFE.
- `tools/plot_fid.py`: FID vs training steps and vs GPU hours from the `results/*/fid/*.json` files,
  PDF + PNG + CSV, and the speed-up at equal FID / FID gap at equal time.

## Not tested here
Tested on CPU only, with fake latents, under torch 2.5.1 / timm 1.0.30 / diffusers 0.35.2 /
Python 3.12: train, kill and resume, 2-process DDP for every variant (baseline, V1, V1 + aux,
V2, V3), fp16 and bf16 autocast (CPU autocast), `torch.compile` (CPU backend), checkpoint loading
in the samplers, and `scripts/train.sh` end to end with a simulated time-limit signal.

Not run: real GPUs (CUDA autocast, GradScaler on real fp16 gradients, throughput), real Slurm,
and the Hugging Face download in `tools/encode_latents.py`.

## Token budgets (nested queries)

One model, several bottleneck sizes. Old checkpoints and old commands behave exactly as before.

- `models.py`: the V1 compressor takes `num_queries=K` (first K queries + a K-cell pooled grid).
  `SiT(query_budgets=(8,16,32,64), budget_mode=...)` trains all budgets in one run: each batch is
  split into equal groups, one per budget. `uniform` = random groups, `time` = ranked by t so the
  noisiest samples get the fewest tokens, `time_rev` = the opposite (control).
  `model.set_budget(...)` picks the budget at sampling; `forward_with_weak_guidance` guides the
  model with a smaller budget of itself.
- `train_utils.py`: `--query-budgets 8,16,32,64 --budget-mode {uniform,time,time_rev} --budget-jitter`.
  Stored in the checkpoint, so eval rebuilds the right model.
- `eval_fid.py`: `--num-queries 32` or a schedule `--num-queries 0.34:8,0.5:16,0.66:32,1:64`
  (t=0 noise, t=1 data), `--weak-queries 16 --guidance-scale 1.5`. Results go to
  `results/<exp>/fid_budget/`, separate from `fid/`.
- `tools/budget_table.py`: prints all `fid_budget` results as a table.

Train (S/2):

    EXP=nested_uniform sbatch --job-name=nest_uni  --exclude=gpu02,gpu04 scripts/train.sh method --query-budgets 8,16,32,64 --budget-mode uniform --ckpt-every 25000
    EXP=nested_time    sbatch --job-name=nest_time --exclude=gpu02,gpu04 scripts/train.sh method --query-budgets 8,16,32,64 --budget-mode time --ckpt-every 25000
