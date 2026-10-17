"""
Paper figure: FID against training steps (a) and against training time (b), baseline vs method.
Reads the JSON files that eval_fid.py writes to results/<run>/fid/, so it can be rerun at any time
as more checkpoints get scored.

    python plot_fid.py                                   # S/2: results/baseline vs results/method
    python plot_fid.py --run "SiT-B/2=results/baseline_B2" --run "SiT-B/2 + ours=results/method_B2" \\
        --speed 2.6 --speed 4.4 --name fid_B2

--speed is training iterations per second of each run (same order as --run). The defaults are the
values in the S/2 training logs (both runs on gpu03, A100 PCIe 40GB). Use numbers measured on one
GPU type for all runs (bench_cost.py), otherwise panel (b) compares hardware, not methods.

Training time is steps / it_s: steady-state GPU hours without checkpointing or data-loading stalls
(the S/2 jobs took 16.4 h and 9.7 h of wall clock; the ratio is the same).

Output: results/figures/<name>.pdf (vector, for LaTeX), .png (preview) and .csv (the plotted points),
plus a printed table with the speed-up at equal FID and the FID gap at equal training time.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

# validated categorical slots 1-2 (light surface, CVD-safe; see the dataviz palette)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def load_run(folder, args):
    pts = {}
    for path in glob.glob(os.path.join(folder, "fid", "*.json")):
        if os.path.basename(path).startswith("smoke_"):
            continue
        r = json.load(open(path))
        if (r.get("num_samples") != args.num_samples or r.get("mode") != args.mode
                or r.get("num_sampling_steps") != args.sampling_steps or float(r.get("cfg_scale", 1.0)) != args.cfg):
            continue
        pts[int(r["train_steps"])] = float(r["fid"])
    steps = np.array(sorted(pts))
    return steps, np.array([pts[s] for s in steps])


def hours_to_reach(target_fid, hours, fids):
    """Training hours a run needs to reach target_fid (linear interpolation; None if never reached)."""
    for i in range(1, len(fids)):
        if fids[i] <= target_fid <= fids[i - 1]:
            f0, f1 = fids[i - 1], fids[i]
            w = 0.0 if f0 == f1 else (f0 - target_fid) / (f0 - f1)
            return hours[i - 1] + w * (hours[i] - hours[i - 1])
    if len(fids) and fids[0] <= target_fid:
        return None   # reached before the first scored checkpoint: no bound from the data
    return None


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = args.run or ["SiT-S/2=results/baseline", "SiT-S/2 + ours=results/method"]
    speeds = args.speed or [7.75, 13.29]
    assert len(speeds) == len(runs), "give one --speed per --run"
    data = []
    for spec, it_s in zip(runs, speeds):
        label, folder = spec.split("=", 1)
        steps, fids = load_run(folder, args)
        assert len(steps), f"no FID results matching the filters in {folder}/fid"
        data.append(dict(label=label, steps=steps, fids=fids, hours=steps / it_s / 3600, it_s=it_s))

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
                         "xtick.color": "#5c5c57", "ytick.color": "#5c5c57", "text.color": "#21211f",
                         "axes.labelcolor": "#21211f"})
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.width * 0.38), sharey=True)
    metric = f"FID-{args.num_samples // 1000}K"
    for ax, key, xlabel in ((axes[0], "steps", "Training steps"), (axes[1], "hours", "Training time (GPU hours)")):
        ends = []
        for i, d in enumerate(data):
            x = d[key] / 1000 if key == "steps" else d[key]
            ax.plot(x, d["fids"], color=COLORS[i], lw=2, marker=MARKERS[i], ms=5, mec="white", mew=1.0,
                    label=d["label"], zorder=3)
            ends.append([x[-1], d["fids"][-1]])
        # end-of-line value labels, pushed apart vertically when two runs end at almost the same FID
        lo, hi = min(d["fids"].min() for d in data), max(d["fids"].max() for d in data)
        gap = 0.06 * (hi - lo)
        order = sorted(range(len(ends)), key=lambda i: ends[i][1])
        ys = [ends[i][1] for i in order]
        xspan = max(e[0] for e in ends) - min(min(d[key]) / (1000 if key == "steps" else 1) for d in data)
        for k in range(1, len(ys)):
            if abs(ends[order[k]][0] - ends[order[k - 1]][0]) < 0.08 * xspan:   # only when they end side by side
                ys[k] = max(ys[k], ys[k - 1] + gap)
        for k, i in enumerate(order):
            ax.annotate(f"{ends[i][1]:.1f}", (ends[i][0], ys[k]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color="#3d3d3a", annotation_clip=False)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e4e4e0", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        if key == "steps":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}K"))
    axes[0].set_ylabel(metric)
    axes[0].set_title("(a) FID vs. training steps", loc="left")
    axes[1].set_title("(b) FID vs. training time", loc="left")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.5)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "train_steps", "gpu_hours", metric])
        for d in data:
            for s, h, f in zip(d["steps"], d["hours"], d["fids"]):
                w.writerow([d["label"], int(s), round(float(h), 3), round(float(f), 3)])
    print(f"saved {base}.pdf / .png / .csv")

    # numbers for the text: each later run against the first one (the baseline)
    ref = data[0]
    for d in data[1:]:
        print(f"\n{d['label']} vs {ref['label']}")
        print(f"{'step':>8} {'hours':>7} {metric:>9} | {'base FID at same hours':>22} | {'base hours to same FID':>22} {'speed-up':>8}")
        for s, h, f in zip(d["steps"], d["hours"], d["fids"]):
            base_f = np.interp(h, ref["hours"], ref["fids"]) if ref["hours"][0] <= h <= ref["hours"][-1] else None
            hb = hours_to_reach(f, ref["hours"], ref["fids"])
            print(f"{s:>8} {h:>7.2f} {f:>9.2f} | {('%.2f' % base_f) if base_f is not None else 'out of range':>22} | "
                  f"{('%.2f' % hb) if hb is not None else 'not reached':>22} {('%.2fx' % (hb / h)) if hb else '-':>8}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", help='"Label=results/<run folder>"; first one is the baseline.')
    p.add_argument("--speed", action="append", type=float, help="Training it/s of each run, same order as --run.")
    p.add_argument("--num-samples", type=int, default=50000)
    p.add_argument("--mode", default="SDE")
    p.add_argument("--sampling-steps", type=int, default=250)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--width", type=float, default=7.0, help="Figure width in inches (7 = two-column text width).")
    p.add_argument("--out-dir", default="results/figures")
    p.add_argument("--name", default="fid_S2")
    main(p.parse_args())
