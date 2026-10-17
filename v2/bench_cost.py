"""
Cost table for the paper: baseline (plain SiT) vs method (token compressor), same GPU, same settings.

For every model size and variant it reports
  * parameters (all, and without the unused rows of the decompressor's position-query bank)
  * forward GFLOPs per image, DiT convention (1 multiply-add = 1 FLOP, so SiT-S/2 = 6.06)
  * training: iterations/s, images/s and peak GPU memory for a real train.py step
    (loss + backward + AdamW + EMA, autocast, optional torch.compile), synthetic latents
  * sampling: model forwards per second and images/s for a 250-step sampler (NFE = 250)

    python bench_cost.py                                    # S/2 and B/2, baseline and method
    python bench_cost.py --models SiT-XL/2 --batch-size 32  # one size, smaller batch
    sbatch --exclude=gpu04 bench.sh                         # on the cluster (see bench.sh)

No data, VAE or checkpoint is needed. Numbers are per GPU, without DDP (train.py also wraps
a single GPU in DDP, so its logged it/s is a few percent lower).
"""
import argparse
import json
import os
import time

import torch
import torch._dynamo

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models import SiT_models
from transport import create_transport

VARIANTS = {
    # same flags as run.sh: baseline = --no-compress --no-detailer, method = defaults
    "baseline": dict(compress=False, detailer=False),
    "method": dict(compress=True, detailer=True),
}


def build(model_name, variant, args):
    kw = dict(input_size=args.image_size // 8, num_classes=1000, query_seqlen=args.query_len,
              compressor_version=args.compressor_version, legacy_residual=False, zero_init_final=False,
              **VARIANTS[variant])
    return SiT_models[model_name](**kw)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    unused = 0
    if getattr(model, "compress", False):
        bank = model.decompressor.pos_query                        # (max_seqlen, D), only L rows are read
        unused = (bank.shape[0] - model.x_embedder.num_patches) * bank.shape[1]
    return total, total - unused


def forward_gflops(model, device, image_size):
    from torch.utils.flop_counter import FlopCounterMode
    lat = image_size // 8
    x = torch.randn(1, 4, lat, lat, device=device)
    t = torch.rand(1, device=device)
    y = torch.zeros(1, dtype=torch.long, device=device)
    model.eval().requires_grad_(False)   # (no_grad() trips the counter's module tracker in torch 2.5)
    # math attention backend: attention becomes plain matmuls, which the counter always sees
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        model(x, t, y)
    return fc.get_total_flops() / 2 / 1e9                          # MACs, the DiT/fvcore convention


@torch.no_grad()
def update_ema(ema, model, decay=0.9999):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(p.detach(), alpha=1 - decay)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_train(model_name, variant, args, device, amp_dtype):
    from copy import deepcopy
    torch.manual_seed(0)
    raw = build(model_name, variant, args).to(device)
    ema = deepcopy(raw).requires_grad_(False)
    model = raw
    if args.compile and device.type == "cuda":
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(raw)
    transport = create_transport("Linear", "velocity", None, None, None)
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-4, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    lat = args.image_size // 8
    B = args.batch_size
    x = torch.randn(B, 4, lat, lat, device=device)
    y = torch.randint(0, 1000, (B,), device=device)
    raw.train()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = transport.training_losses(model, x, dict(y=y))["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1e9)
        scaler.step(opt)
        scaler.update()
        update_ema(ema, raw)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.warmup):                                   # includes compilation
        step()
    sync(device)
    t0 = time.time()
    for _ in range(args.steps):
        step()
    sync(device)
    dt = (time.time() - t0) / args.steps
    mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    del model, raw, ema, opt
    return {"train_it_s": 1 / dt, "train_img_s": B / dt, "train_peak_mem_gb": mem}


@torch.no_grad()
def bench_sample(model_name, variant, args, device, dtype):
    torch.manual_seed(0)
    model = build(model_name, variant, args).to(device).eval()
    lat = args.image_size // 8
    B = args.sample_batch_size
    x = torch.randn(B, 4, lat, lat, device=device)
    t = torch.rand(B, device=device)
    y = torch.randint(0, 1000, (B,), device=device)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        for _ in range(args.warmup):
            model(x, t, y)
        sync(device)
        t0 = time.time()
        for _ in range(args.steps):
            model(x, t, y)
        sync(device)
    dt = (time.time() - t0) / args.steps
    del model
    return {"fwd_img_s": B / dt, "img_s_250nfe": B / dt / 250}


def main(args):
    use_cuda = torch.cuda.is_available()
    assert use_cuda or args.allow_cpu, "needs a GPU (--allow-cpu only for tests)"
    device = torch.device("cuda" if use_cuda else "cpu")
    gpu = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    if args.amp == "auto":
        amp_dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if use_cuda else None
    else:
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp]
    print(f"GPU: {gpu} | train autocast {amp_dtype} | compile {args.compile} | batch {args.batch_size} | "
          f"sampling batch {args.sample_batch_size} (fp32/tf32, as in eval_fid.py)", flush=True)

    rows = []
    for name in args.models:
        for variant in args.variants:
            m = build(name, variant, args)
            total, used = count_params(m)
            row = {"model": name, "variant": variant, "gpu": gpu, "params_m": total / 1e6,
                   "params_used_m": used / 1e6, "gflops_fwd": forward_gflops(m.to(device), device, args.image_size)}
            del m
            for label, fn in (("train", lambda: bench_train(name, variant, args, device, amp_dtype)),
                              ("sample", lambda: bench_sample(name, variant, args, device, None))):
                try:
                    row.update(fn())
                except torch.cuda.OutOfMemoryError:
                    row[f"{label}_error"] = "OOM"
                if use_cuda:
                    torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)

    print("\n| Model | Variant | Params (M) | GFLOPs/img | Train it/s | Train img/s | Peak mem (GB) | Sample img/s (250 NFE) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        def f(k, fmt, label):
            return fmt.format(r[k]) if k in r else r.get(f"{label}_error", "-")
        print(f"| {r['model']} | {r['variant']} | {r['params_m']:.2f} ({r['params_used_m']:.2f} used) | "
              f"{r['gflops_fwd']:.2f} | {f('train_it_s', '{:.2f}', 'train')} | {f('train_img_s', '{:.0f}', 'train')} | "
              f"{f('train_peak_mem_gb', '{:.1f}', 'train')} | {f('img_s_250nfe', '{:.2f}', 'sample')} |")
    for name in args.models:
        b = next((r for r in rows if r["model"] == name and r["variant"] == "baseline"), None)
        m = next((r for r in rows if r["model"] == name and r["variant"] == "method"), None)
        if b and m:
            ratio = lambda k: (f"{m[k] / b[k]:.2f}x" if k in b and k in m else "-")
            print(f"{name}: method/baseline  train speed {ratio('train_it_s')}, sampling speed {ratio('img_s_250nfe')}, "
                  f"FLOPs {m['gflops_fwd'] / b['gflops_fwd']:.2f}x")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["SiT-S/2", "SiT-B/2"], choices=list(SiT_models.keys()))
    p.add_argument("--variants", nargs="+", default=["baseline", "method"], choices=list(VARIANTS))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--query-len", type=int, default=32)
    p.add_argument("--compressor-version", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--batch-size", type=int, default=256, help="Training batch on this one GPU (run.sh uses 256).")
    p.add_argument("--sample-batch-size", type=int, default=250, help="eval_fid.py uses 250 per GPU.")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", type=str, default=None, help="JSON file for the rows.")
    p.add_argument("--allow-cpu", action="store_true")
    main(p.parse_args())
