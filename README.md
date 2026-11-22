# SiT + token compression

SiT (flow-matching transformer, ImageNet 256×256 latents) with a Q-Former–style token bottleneck:
the first and last `depth/6` blocks run on all 256 tokens; the middle blocks run on 32 compressed
tokens (pool branch + learned queries), which a Perceiver-IO decompressor maps back to 256 tokens.
Optional nested query budgets (8/16/32/64) train one model for several bottleneck sizes.
Details of every change from upstream SiT: [docs/CHANGES.md](docs/CHANGES.md).

## Layout

```
.                         repository root = /media/ltnghia33/sit on the cluster
├── models.py             SiT, compressor (V1/V2/V3), decompressor, token budgets
├── transport/            flow-matching interpolants, losses, ODE/SDE samplers (upstream SiT)
├── dataset.py            MemmapLatentDataset (data/latents.npy + labels.npy)
├── train.py              training (DDP, resume, AMP, EMA milestones)
├── eval_fid.py           FID-50K in memory, one JSON per checkpoint
├── sample.py, sample_ddp.py   image samples / sample folders
├── bench_cost.py         params, GFLOPs, train/sample speed, memory
├── paths.py              every default folder (Python side)
├── train_utils.py, download.py, wandb_utils.py
├── scripts/              Slurm and shell entry points
│   ├── env.sh            THE ONLY FILE WITH MACHINE-SPECIFIC PATHS (conda, data, ...)
│   ├── train.sh          sbatch scripts/train.sh {baseline|method|legacy} [train.py flags]
│   ├── fid.sh            sbatch scripts/fid.sh <ckpt.pt> ...
│   ├── bench.sh          cost table
│   ├── prepare_fid.sh    one-time: FID reference stats, Inception, SD-VAE -> fid_ref/
│   ├── encode.sh         one-time: ImageNet -> data/latents.npy, data/labels.npy
│   └── migrate_layout.sh one-time: old v2/ layout -> this one
├── tools/                no GPU needed
│   ├── plot_fid.py, plot_loss.py, plot_efficiency.py, budget_table.py   paper figures / tables
│   └── encode_latents.py, download_shards.py, repack_latents.py         data preparation
├── docs/CHANGES.md
│
│   not in git (see .gitignore):
├── data/                 latents.npy (42GB), labels.npy
├── fid_ref/              VIRTUAL_imagenet256_labeled_stats.npz, torch/ (Inception), sd-vae-ft-ema/
├── results/<exp>/        checkpoints/, fid/, fid_budget/, log.txt     results/figures/, results/bench/
├── logs/                 Slurm .out / .err, one subfolder per kind of job (logs/README.md)
└── pretrained_models/    released SiT-XL/2 weights (only for sample.py without --ckpt)
```

**Every command is run from the repository root.** Python defaults come from `paths.py`, shell
defaults from `scripts/env.sh`; both resolve folders relative to the repository, so nothing breaks
when the repository is cloned somewhere else. To put data or results elsewhere, set
`DATA=`, `RESULTS=`, `FID_REF=` (shell) or `SIT_DATA`, `SIT_RESULTS`, `SIT_FID_REF` (Python).

## Setup

```bash
conda create -n sit python=3.12 -y && conda activate sit
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt          # diffusers must stay at 0.35.2 with torch 2.5
```

On a new machine, edit `CONDA_SH` (and `PARQUET` if you encode data) in `scripts/env.sh`.

One-time data (skip what already exists):

```bash
sbatch scripts/encode.sh                  # -> data/latents.npy, data/labels.npy
bash scripts/prepare_fid.sh               # login node, needs internet -> fid_ref/
```

## Experiments

```bash
cd /media/ltnghia33/sit                   # repository root

# training (resubmitting the same command resumes from results/<EXP>/checkpoints/latest.pt)
sbatch --job-name=base   scripts/train.sh baseline
sbatch --job-name=method scripts/train.sh method
MODEL=SiT-B/2 EXP=method_B2 sbatch scripts/train.sh method
EXP=nested_time sbatch scripts/train.sh method --query-budgets 8,16,32,64 --budget-mode time --ckpt-every 25000
DRY_RUN=1 bash scripts/train.sh method    # print the command only
SMOKE=1  bash scripts/train.sh method     # 200 steps, inside an interactive GPU session

# FID-50K (JSON in results/<exp>/fid/)
sbatch scripts/fid.sh results/method/checkpoints/0*.pt
FID_ARGS="--num-samples 10000 --num-queries 32" sbatch scripts/fid.sh results/nested_time/checkpoints/0400000.pt

# cost table, figures, tables
sbatch --nodelist=gpu03 scripts/bench.sh
python tools/plot_fid.py
python tools/plot_loss.py --run "SiT-S/2=logs/train/base_78522.err" --run "SiT-S/2 + ours=logs/train/method_78588.err" --name loss_S2
python tools/plot_efficiency.py
python tools/budget_table.py
```

## Moving an old copy (code in `v2/`) to this layout

No job may be running. On the cluster:

```bash
cd /media/ltnghia33/sit
git pull
bash scripts/migrate_layout.sh            # dry run: lists every move
APPLY=1 bash scripts/migrate_layout.sh    # latents/labels -> data/, v2/results -> results/, v2/logs -> logs/
```

Checkpoints and FID JSON files need no change: runs are found by folder name, and the absolute
paths stored inside old JSON files are only metadata.

## Credits

Built on [SiT](https://github.com/willisma/SiT) (Ma et al., 2024), MIT license (LICENSE.txt).
