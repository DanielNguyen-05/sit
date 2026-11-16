"""
Token-budget analysis without sampling or FID (step 1c). For every checkpoint, on a fixed subset of
the training latents, at each noise level t and each bottleneck budget K it measures

  L(t, K)        the training loss (flow matching, velocity MSE), with the SAME data, labels and
                 noise x0 for every K and every checkpoint, so differences are paired and precise
  dL(t, K)       L(t, K) - L(t, Kmax): what a smaller budget costs at that noise level. The
                 irreducible part of the loss cancels, so this is the extra error of budget K.
  d(K, K')       RMS of v_K - v_K': how much the prediction changes between consecutive budgets
  cos            cos(v_K2 - v_K1, v_K3 - v_K2): do consecutive corrections point the same way?
                 (high = the "weak model errs like the strong one, only more" premise of
                 autoguidance holds, so extrapolating from a smaller budget is justified)
  alpha(t)       decay rate of the budget error, from d(K1,K2) / d(K2,K3) = (K2/K1)^alpha
  s(t)           guidance scale that extrapolates weak + s (full - weak) to the infinite-budget
                 limit if the error decays as K^-alpha: s = r^alpha / (r^alpha - 1), r = Kfull/Kweak

Checkpoints without budgets (baseline, single-budget method) get L(t) only, on the same samples, so
every model can be compared at every t: e.g. nested K=32 vs the method trained at 32 vs baseline.

    python analyze_budgets.py --ckpt results/nested_uniform/checkpoints/0400000.pt \\
        results/nested_time/checkpoints/0400000.pt results/method/checkpoints/0400000.pt \\
        results/baseline/checkpoints/0400000.pt
    sbatch scripts/budget_loss.sh --ckpt ...          # on the cluster

Output: results/<exp>/budget_loss/<step>_n<N>.json per checkpoint, results/figures/<name>.csv/.png/.pdf
and a printed table. About 5-15 min on one A100 for four S/2 checkpoints at the defaults.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

import paths
from models import SiT_models
from train_utils import apply_checkpoint_args, model_kwargs_from_args, parse_model_args, parse_transport_args
from transport import create_transport

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ----------------------------------------------------------------------------------------- data
def load_subset(data_path, n, seed, log):
    """Fixed random subset of the training latents, sampled exactly as train.py does
    (posterior sample of the SD-VAE latent, times 0.18215), plus fixed noise x0."""
    if not os.path.isfile(os.path.join(data_path, "latents.npy")):
        alt = paths.REPO
        assert os.path.isfile(os.path.join(alt, "latents.npy")), \
            f"latents.npy not found in {data_path} (nor in {alt}); pass --data-path"
        log(f"[data] {data_path}/latents.npy missing, using {alt} (old layout; run scripts/migrate_layout.sh)")
        data_path = alt
    lat = np.load(os.path.join(data_path, "latents.npy"), mmap_mode="r")
    lab = np.load(os.path.join(data_path, "labels.npy"), mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(lab), size=n, replace=False))      # sorted: sequential reads on the memmap
    t0 = time.time()
    moments = torch.from_numpy(np.asarray(lat[idx], dtype=np.float32))   # (n, 8, h, w): mean | logvar
    y = torch.from_numpy(np.asarray(lab[idx], dtype=np.int64))
    log(f"[data] {n} latents read in {time.time() - t0:.0f}s from {data_path}")
    gen = torch.Generator().manual_seed(seed)
    mean, logvar = moments.chunk(2, dim=1)
    std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))               # as diffusers' DiagonalGaussianDistribution
    x1 = (mean + std * torch.randn(mean.shape, generator=gen)) * 0.18215
    x0 = torch.randn(x1.shape, generator=gen)
    return x1, x0, y


# ---------------------------------------------------------------------------------------- model
def load_model(ckpt, args, device, log):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert isinstance(raw, dict) and "ema" in raw, f"{ckpt}: expected a train.py checkpoint (with 'ema')"
    margs = argparse.Namespace(**vars(args))
    apply_checkpoint_args(margs, raw.get("args"), log=log)
    model = SiT_models[margs.model](learn_sigma=False, **model_kwargs_from_args(margs)).to(device)
    model.load_state_dict(raw["ema"])
    model.eval()
    steps = raw.get("train_steps")
    del raw
    if not margs.compress:
        budgets = [None]                                     # plain SiT
    elif model.query_budgets:
        budgets = list(model.query_budgets)                  # nested: every trained budget
    elif hasattr(model.compressor, "query"):
        budgets = [int(model.compressor.query.shape[0])]     # single-budget V1 compressor
    else:
        budgets = [None]                                     # V2 / V3: no budget knob
    return model, margs, steps, budgets


def run_name(ckpt):
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))))   # results/<exp>/checkpoints/x.pt


# -------------------------------------------------------------------------------------- measure
@torch.no_grad()
def measure(model, budgets, transport, x1, x0, y, tgrid, args, device, log):
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    nK = len(budgets)
    pairs = list(zip(budgets[:-1], budgets[1:]))
    triples = list(zip(budgets[:-2], budgets[1:-1], budgets[2:]))
    rows = []
    for ti, t in enumerate(tgrid.tolist()):
        loss = [[] for _ in range(nK)]          # per-sample losses, per budget
        pdiff = [[] for _ in pairs]             # per-sample mean squared differences of consecutive budgets
        cosv = [[] for _ in triples]
        for b in range(0, x1.shape[0], args.batch_size):
            x1b = x1[b:b + args.batch_size].to(device)
            x0b = x0[b:b + args.batch_size].to(device)
            yb = y[b:b + args.batch_size].to(device)
            tt = torch.full((x1b.shape[0],), t, device=device)
            _, xt, ut = transport.path_sampler.plan(tt, x0b, x1b)
            preds = []
            for k, K in enumerate(budgets):
                with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                    v = model(xt, tt, yb) if K is None else model(xt, tt, yb, num_queries=K)
                v = v.float()
                preds.append(v)
                loss[k].append(((v - ut) ** 2).mean(dim=(1, 2, 3)).cpu())
            for p, (a, c) in enumerate(pairs):
                ia, ic = budgets.index(a), budgets.index(c)
                pdiff[p].append(((preds[ic] - preds[ia]) ** 2).mean(dim=(1, 2, 3)).cpu())
            for q, (a, m, c) in enumerate(triples):
                ia, im, ic = budgets.index(a), budgets.index(m), budgets.index(c)
                d1 = (preds[im] - preds[ia]).flatten(1)
                d2 = (preds[ic] - preds[im]).flatten(1)
                cosv[q].append(torch.nn.functional.cosine_similarity(d1, d2, dim=1).cpu())
        loss = [torch.cat(l).double() for l in loss]
        row = {"t": t, "loss": {}, "loss_se": {}, "dloss": {}, "dloss_se": {}, "rms_diff": {}, "cos": {}}
        for k, K in enumerate(budgets):
            row["loss"][str(K)] = loss[k].mean().item()
            row["loss_se"][str(K)] = (loss[k].std() / math.sqrt(len(loss[k]))).item()
            dl = loss[k] - loss[-1]                                            # paired: same samples, same noise
            row["dloss"][str(K)] = dl.mean().item()
            row["dloss_se"][str(K)] = (dl.std() / math.sqrt(len(dl))).item() if k < nK - 1 else 0.0
        for p, (a, c) in enumerate(pairs):
            row["rms_diff"][f"{a}-{c}"] = math.sqrt(torch.cat(pdiff[p]).double().mean().item())
        for q, (a, m, c) in enumerate(triples):
            row["cos"][f"{a}-{m}-{c}"] = torch.cat(cosv[q]).double().mean().item()
        # decay exponent from the top triple (largest budgets), and the extrapolating guidance scale
        row["alpha"], row["scale"] = None, {}
        if triples:
            a, m, c = triples[-1]
            d_lo, d_hi = row["rms_diff"][f"{a}-{m}"], row["rms_diff"][f"{m}-{c}"]
            if d_lo > 0 and d_hi > 0:
                row["alpha"] = math.log(d_lo / d_hi) / math.log(m / a)
            for Kw in budgets[:-1]:
                r = budgets[-1] / Kw
                al = row["alpha"]
                row["scale"][str(Kw)] = (r ** al / (r ** al - 1)) if (al is not None and al > 0) else None
        rows.append(row)
        if ti % max(1, len(tgrid) // 5) == 0:
            log(f"  t={t:.3f}  " + "  ".join(f"L[{K}]={row['loss'][str(K)]:.4f}" for K in budgets))
    return rows


def train_weighted(rows, budgets):
    """Loss averaged with train.py's t distribution (logit-normal: t = sigmoid(N(0,1)))."""
    ts = np.array([r["t"] for r in rows])
    w = np.exp(-0.5 * np.log(ts / (1 - ts)) ** 2) / (ts * (1 - ts))
    w = w / w.sum()
    return {str(K): float(sum(wi * r["loss"][str(K)] for wi, r in zip(w, rows))) for K in budgets}


# ------------------------------------------------------------------------------------- reporting
def print_table(name, budgets, rows, log):
    Kmax = budgets[-1]
    head = f"{'t':>6} " + " ".join(f"{'L' + str(K):>9}" for K in budgets)
    if len(budgets) > 1:
        head += " " + " ".join(f"{'dL' + str(K):>9}" for K in budgets[:-1])
        head += " " + " ".join(f"{'cos' + k.split('-')[1]:>8}" for k in rows[0]["cos"])
        head += f" {'alpha':>6} " + " ".join(f"{'s(' + str(Kmax) + '/' + str(K) + ')':>9}" for K in budgets[:-1])
    log(f"\n== {name}   budgets {budgets}")
    log(head)
    for r in rows:
        line = f"{r['t']:6.3f} " + " ".join(f"{r['loss'][str(K)]:9.4f}" for K in budgets)
        if len(budgets) > 1:
            line += " " + " ".join(f"{r['dloss'][str(K)]:9.5f}" for K in budgets[:-1])
            line += " " + " ".join(f"{v:8.3f}" for v in r["cos"].values())
            line += f" {r['alpha']:6.2f}" if r["alpha"] is not None else f" {'-':>6}"
            line += " " + " ".join(f"{v:9.2f}" if v is not None else f"{'-':>9}" for v in r["scale"].values())
        log(line)


def write_csv(path, results):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "steps", "t", "K", "loss", "loss_se", "dloss_vs_Kmax", "dloss_se", "alpha"])
        for res in results:
            for r in res["rows"]:
                for K in res["budgets"]:
                    w.writerow([res["run"], res["train_steps"], f"{r['t']:.4f}", K, f"{r['loss'][str(K)]:.6f}",
                                f"{r['loss_se'][str(K)]:.6f}", f"{r['dloss'][str(K)]:.6f}",
                                f"{r['dloss_se'][str(K)]:.6f}", "" if r["alpha"] is None else f"{r['alpha']:.4f}"])


def plot(results, base, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib missing: no figure (the CSV has everything)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ref = next((r for r in results if r["budgets"] == [None]), None)       # a plain SiT, if given
    for res in results:
        ts = [r["t"] for r in res["rows"]]
        for K in res["budgets"]:
            label = res["run"] + ("" if K is None else f" K={K}")
            if ref is not None and res is not ref:
                y = [r["loss"][str(K)] - rr["loss"]["None"] for r, rr in zip(res["rows"], ref["rows"])]
                ax[0].plot(ts, y, label=label)
            if len(res["budgets"]) > 1 and K != res["budgets"][-1]:
                ax[1].plot(ts, [r["dloss"][str(K)] for r in res["rows"]], label=label)
        if len(res["budgets"]) > 2:
            ax[2].plot(ts, [r["alpha"] if r["alpha"] is not None else np.nan for r in res["rows"]],
                       label=res["run"] + " alpha")
            ax[2].plot(ts, [r["cos"][list(r["cos"])[-1]] for r in res["rows"]], "--",
                       label=res["run"] + f" cos {list(res['rows'][0]['cos'])[-1]}")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_title(f"loss minus {ref['run']} (paired)" if ref else "(no plain SiT given)")
    ax[1].set_title("dL(t, K) = L(K) - L(Kmax)")
    ax[2].set_title("alpha(t) and cos of consecutive corrections")
    for a in ax:
        a.set_xlabel("t   (0 = noise, 1 = data)")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=150)
    log(f"figure -> {base}.png / .pdf")


# ------------------------------------------------------------------------------------------ main
def main(args):
    use_cuda = torch.cuda.is_available()
    assert use_cuda or args.allow_cpu, "needs a GPU (--allow-cpu for tests only)"
    device = torch.device("cuda" if use_cuda else "cpu")
    log = lambda *a: print(*a, flush=True)
    assert args.prediction == "velocity", "this analysis assumes a velocity model"

    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    x1, x0, y = load_subset(args.data_path, args.num_samples, args.seed, log)
    tgrid = torch.linspace(args.t_min, args.t_max, args.num_t)

    results = []
    for ckpt in args.ckpt:
        assert os.path.isfile(ckpt), f"checkpoint not found: {ckpt}"
        t0 = time.time()
        log(f"=== {ckpt}")
        model, margs, steps, budgets = load_model(ckpt, args, device, log)
        rows = measure(model, budgets, transport, x1, x0, y, tgrid, args, device, log)
        res = {"ckpt": os.path.abspath(ckpt), "run": run_name(ckpt), "train_steps": steps, "model": margs.model,
               "budgets": budgets, "budget_mode": getattr(margs, "budget_mode", None) if len(budgets) > 1 else None,
               "num_samples": args.num_samples, "seed": args.seed, "amp": args.amp,
               "train_weighted_loss": train_weighted(rows, budgets), "rows": rows,
               "minutes": round((time.time() - t0) / 60, 1)}
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt))), "budget_loss",
                           f"{os.path.splitext(os.path.basename(ckpt))[0]}_n{args.num_samples}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print_table(f"{res['run']} @ {steps}", budgets, rows, log)
        log("train-weighted loss: " + "  ".join(f"K={k}: {v:.5f}" for k, v in res["train_weighted_loss"].items()))
        log(f"-> {out}  ({res['minutes']} min)")
        results.append(res)
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    write_csv(base + ".csv", results)
    log(f"csv -> {base}.csv")
    plot(results, base, log)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--ckpt", type=str, nargs="+", required=True, help="train.py checkpoints (EMA weights are used).")
    p.add_argument("--data-path", type=str, default=paths.DATA_DIR, help="Folder with latents.npy + labels.npy.")
    p.add_argument("--num-samples", type=int, default=4096, help="Same subset for every checkpoint.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-t", type=int, default=25, help="Noise levels, evenly spaced in [t-min, t-max].")
    p.add_argument("--t-min", type=float, default=0.02)
    p.add_argument("--t-max", type=float, default=0.98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", type=str, choices=["none", "bf16", "fp16"], default="none",
                   help="none (fp32/tf32) keeps the small loss differences exact.")
    p.add_argument("--out-dir", type=str, default=paths.FIGURES_DIR)
    p.add_argument("--name", type=str, default="budget_loss")
    p.add_argument("--allow-cpu", action="store_true", help="Tests only.")
    # architecture fallbacks for checkpoints without stored args (train.py checkpoints carry their own)
    p.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/2")
    p.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    p.add_argument("--num-classes", type=int, default=1000)
    parse_model_args(p)
    parse_transport_args(p)
    main(p.parse_args())
