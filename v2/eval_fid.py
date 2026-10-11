"""
FID-50K for SiT checkpoints, computed in memory: samples are generated, decoded, passed through
Inception-v3 and reduced to running sums on the GPU. No image is written to disk; the output is a
small JSON per checkpoint.

    torchrun --standalone --nproc_per_node=1 eval_fid.py \
        --ckpt results/baseline/checkpoints/0400000.pt results/method/checkpoints/0400000.pt \
        --ref-stats /media/ltnghia33/sit/fid_ref/VIRTUAL_imagenet256_labeled_stats.npz

Defaults follow the SiT paper's 400K-step numbers (SiT-S/2 = 57.6): SDE Euler-Maruyama, 250 steps,
no classifier-free guidance, 50,000 samples, class labels drawn uniformly, SD-VAE ft-EMA decoder,
images rounded to uint8 exactly as sample_ddp.py does before they are scored.

Inception: the pytorch-fid port of the TF Inception used by the ADM evaluator. Its activations match
the TF graph closely but not bit for bit, so absolute values can differ from the ADM suite in the
first decimals. Every model scored with this script is scored the same way, so comparisons hold.

Reference statistics: mu / sigma from OpenAI's ADM ImageNet 256 reference batch
(VIRTUAL_imagenet256_labeled.npz), extracted by prepare_fid.sh.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from download import find_model_and_args
from models import SiT_models
from train_utils import model_kwargs_from_args, apply_checkpoint_args, parse_transport_args
from transport import create_transport, Sampler


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), as in the ADM evaluator and pytorch-fid.
    Written out here because pytorch-fid 0.3.0 calls scipy's sqrtm(disp=...), removed in new scipy."""
    from scipy import linalg
    mu1, mu2 = np.atleast_1d(mu1).astype(np.float64), np.atleast_1d(mu2).astype(np.float64)
    sigma1, sigma2 = np.atleast_2d(sigma1).astype(np.float64), np.atleast_2d(sigma2).astype(np.float64)
    diff = mu1 - mu2

    def _sqrtm(a):
        out = linalg.sqrtm(a)
        return out[0] if isinstance(out, tuple) else out

    covmean = _sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():          # singular product: add a small diagonal, as both references do
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def all_sum(t, world):
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def default_out(ckpt, args):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))      # results/<exp>
    tag = f"{args.mode}-{args.sampling_method}-{args.num_sampling_steps}-cfg{args.cfg_scale}-n{args.num_samples}"
    name = os.path.splitext(os.path.basename(ckpt))[0]
    return os.path.join(run_dir, "fid", f"{name}_{tag}.json")


@torch.no_grad()
def evaluate(ckpt, args, rank, world, device, vae, inception, ref_mu, ref_sigma, log):
    state_dict, ckpt_args = find_model_and_args(ckpt)
    margs = argparse.Namespace(**vars(args))
    apply_checkpoint_args(margs, ckpt_args, log=log if rank == 0 else (lambda *a: None))
    model = SiT_models[margs.model](learn_sigma=False, **model_kwargs_from_args(margs)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    latent_size = margs.image_size // 8
    train_steps = None
    if ckpt_args is not None:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        train_steps = raw.get("train_steps")
        del raw

    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(transport)
    if args.mode == "SDE":
        sample_fn = sampler.sample_sde(sampling_method=args.sampling_method, diffusion_form=args.diffusion_form,
                                       diffusion_norm=args.diffusion_norm, last_step=args.last_step,
                                       last_step_size=args.last_step_size, num_steps=args.num_sampling_steps)
    else:
        sample_fn = sampler.sample_ode(sampling_method=args.sampling_method, num_steps=args.num_sampling_steps,
                                       atol=args.atol, rtol=args.rtol)
    use_cfg = args.cfg_scale > 1.0
    model_fn = model.forward_with_cfg if use_cfg else model.forward

    # Same seeds for every checkpoint, so all models are scored on the same noise and labels
    # (the generator draws z and y; the global seed drives the SDE's own noise).
    torch.manual_seed(args.seed * 1000 + rank)
    gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + rank)
    n = args.per_proc_batch_size
    global_batch = n * world
    iterations = math.ceil(args.num_samples / global_batch)
    D = 2048
    feat_sum = torch.zeros(D, dtype=torch.float64, device=device)
    feat_outer = torch.zeros(D, D, dtype=torch.float64, device=device)
    count = torch.zeros(1, dtype=torch.float64, device=device)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    t0 = time.time()
    for it in range(iterations):
        # keep exactly num_samples overall: drop the tail of the last global batch
        first = it * global_batch + rank * n
        keep = max(0, min(n, args.num_samples - first))
        z = torch.randn(n, 4, latent_size, latent_size, device=device, generator=gen)
        y = torch.randint(0, margs.num_classes, (n,), device=device, generator=gen)
        if keep == 0:
            continue
        z, y = z[:keep], y[:keep]
        if use_cfg:
            z = torch.cat([z, z], 0)
            y = torch.cat([y, torch.full_like(y, margs.num_classes)], 0)
            kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        else:
            kwargs = dict(y=y)
        with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
            samples = sample_fn(z, model_fn, **kwargs)[-1]
        samples = samples.float()
        if use_cfg:
            samples, _ = samples.chunk(2, dim=0)
        for i in range(0, samples.shape[0], args.decode_batch_size):
            img = vae.decode(samples[i:i + args.decode_batch_size] / 0.18215).sample
            img = torch.clamp(127.5 * img + 128.0, 0, 255).to(torch.uint8)    # as written to PNG by sample_ddp.py
            feats = inception(img.float() / 255.0)[0].squeeze(-1).squeeze(-1).double()   # (b, 2048)
            feat_sum += feats.sum(0)
            feat_outer += feats.T @ feats
            count += feats.shape[0]
        if rank == 0 and (it + 1) % args.log_every == 0:
            done = min((it + 1) * global_batch, args.num_samples)
            rate = done / (time.time() - t0)
            log(f"  {done:,}/{args.num_samples:,} samples | {rate:.1f} img/s | about "
                f"{(args.num_samples - done) / max(rate, 1e-9) / 60:.0f} min left")

    all_sum(feat_sum, world); all_sum(feat_outer, world); all_sum(count, world)
    N = int(count.item())
    assert N == args.num_samples, f"scored {N} samples, expected {args.num_samples}"
    mu = (feat_sum / N).cpu().numpy()
    sigma = ((feat_outer - N * torch.outer(feat_sum / N, feat_sum / N)) / (N - 1)).cpu().numpy()
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    minutes = (time.time() - t0) / 60
    result = {"ckpt": os.path.abspath(ckpt), "train_steps": train_steps, "fid": fid, "num_samples": N,
              "model": margs.model, "compress": margs.compress, "detailer": margs.detailer,
              "legacy_residual": margs.legacy_residual, "query_len": margs.query_len,
              "mode": args.mode, "sampling_method": args.sampling_method,
              "num_sampling_steps": args.num_sampling_steps, "diffusion_form": args.diffusion_form,
              "last_step": args.last_step, "last_step_size": args.last_step_size, "cfg_scale": args.cfg_scale,
              "amp": args.amp, "seed": args.seed, "vae": args.vae, "ref_stats": os.path.abspath(args.ref_stats),
              "inception": "pytorch-fid pt_inception-2015-12-05", "world_size": world,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "minutes": round(minutes, 1)}
    if args.save_stats and rank == 0:
        np.savez(os.path.splitext(args._out)[0] + "_stats.npz", mu=mu, sigma=sigma)
    del model
    return result


def main(args):
    rank, world = setup_dist()
    use_cuda = torch.cuda.is_available()
    assert use_cuda or args.allow_cpu, "FID evaluation needs a GPU (use --allow-cpu only for tests)."
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    ref = np.load(args.ref_stats)
    ref_mu, ref_sigma = ref["mu"], ref["sigma"]
    assert ref_mu.shape == (2048,) and ref_sigma.shape == (2048, 2048), "reference stats must be 2048-d mu/sigma"

    from diffusers.models import AutoencoderKL
    from pytorch_fid.inception import InceptionV3
    vae = AutoencoderKL.from_pretrained(args.vae if os.path.isdir(args.vae) else f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True).to(device).eval()

    for ckpt in args.ckpt:
        args._out = args.out if (args.out and len(args.ckpt) == 1) else default_out(ckpt, args)
        if os.path.exists(args._out) and not args.overwrite:
            log(f"skip {ckpt}: {args._out} already exists (--overwrite to redo)")
            continue
        log(f"=== {ckpt}")
        if rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args._out)), exist_ok=True)
        result = evaluate(ckpt, args, rank, world, device, vae, inception, ref_mu, ref_sigma, log)
        if rank == 0:
            with open(args._out, "w") as f:
                json.dump(result, f, indent=2)
            log(f"FID {result['fid']:.2f}  ({result['num_samples']:,} samples, {result['minutes']} min)  -> {args._out}")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, nargs="+", required=True, help="One or more train.py checkpoints (EMA weights are used).")
    p.add_argument("--ref-stats", type=str, required=True, help="npz with mu (2048,) and sigma (2048, 2048).")
    p.add_argument("--out", type=str, default=None, help="Result JSON (single checkpoint only). Default: results/<exp>/fid/<step>_<settings>.json")
    p.add_argument("--num-samples", type=int, default=50_000)
    p.add_argument("--per-proc-batch-size", type=int, default=250)
    p.add_argument("--decode-batch-size", type=int, default=50)
    p.add_argument("--mode", type=str, choices=["SDE", "ODE"], default="SDE")
    p.add_argument("--sampling-method", type=str, default="Euler", help="SDE: Euler | Heun. ODE: any torchdiffeq method, e.g. dopri5, heun2.")
    p.add_argument("--num-sampling-steps", type=int, default=250)
    p.add_argument("--diffusion-form", type=str, default="sigma")
    p.add_argument("--diffusion-norm", type=float, default=1.0)
    p.add_argument("--last-step", type=str, default="Mean")
    p.add_argument("--last-step-size", type=float, default=0.04)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--cfg-scale", type=float, default=1.0, help="1.0 = no guidance (as in the 400K comparison of the SiT paper).")
    p.add_argument("--amp", type=str, choices=["none", "bf16", "fp16"], default="none",
                   help="Autocast for the sampling network only. none (tf32) is the reference setting.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae", type=str, default="ema", help="ema | mse | a local folder")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-stats", action="store_true", help="Also save the generated mu/sigma (about 34MB).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Tests only.")
    # architecture fallbacks for checkpoints without stored args (train.py checkpoints carry their own)
    p.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/2")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    from train_utils import parse_model_args
    parse_model_args(p)
    parse_transport_args(p)
    main(p.parse_args())
