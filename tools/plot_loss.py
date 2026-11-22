"""
Training loss of two or more runs, against training steps (a) and against training time (b),
read straight from the Slurm logs (logs/train/<job>.err, the "(step=...) Train Loss: ..." lines).

    python tools/plot_loss.py --run "SiT-S/2=logs/train/base_78522.err" --run "SiT-S/2 + ours=logs/train/method_78588.err" --name loss_S2
    python tools/plot_loss.py --run "SiT-B/2=logs/train/base_B2_79033.err" --run "SiT-B/2 + ours=logs/train/method_B2_79034.err" --name loss_B2

The loss is logged every 100 steps and is noisy, so it is drawn as a trailing mean over --window steps.
Training time = steps / the run's median logged it/s (steady-state GPU hours; the first epoch reads the
data from the network disk and is much slower, and checkpoint saves are not in the logged it/s).
Override with --speed to use one set of numbers for every figure (e.g. the bench_cost.py ones).

Output: results/figures/<name>.pdf / .png / .csv, and a printed table: loss at equal steps and at
equal training time.
"""
import argparse
import csv
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # tools/ -> repository root
import re

import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LINE = re.compile(r"step=(\d+)\) Train Loss: ([0-9.]+).*?Train Steps/Sec: ([0-9.]+)")


def load_log(path):
    steps, loss, speed = {}, {}, {}
    for line in open(path, errors="ignore"):
        m = LINE.search(line)
        if m:   # a resubmitted run repeats some steps; the later line wins
            s = int(m[1]); steps[s] = s; loss[s] = float(m[2]); speed[s] = float(m[3])
    s = np.array(sorted(steps))
    return s, np.array([loss[k] for k in s]), np.array([speed[k] for k in s])


def trailing_mean(steps, values, window):
    c = np.concatenate([[0.0], np.cumsum(values)])
    lo = np.searchsorted(steps, steps - window, side="right")
    hi = np.arange(1, len(steps) + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def main(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert args.run and len(args.run) >= 1, 'give --run "Label=path/to/log.err" at least once'
    assert not args.speed or len(args.speed) == len(args.run), "give one --speed per --run"
    data = []
    for i, spec in enumerate(args.run):
        label, path = spec.split("=", 1)
        s, l, v = load_log(path)
        assert len(s), f"no 'Train Loss' lines in {path}"
        it_s = args.speed[i] if args.speed else float(np.median(v))
        sm = trailing_mean(s, l, args.window)
        keep = s >= args.start
        data.append(dict(label=label, steps=s[keep], loss=sm[keep], hours=s[keep] / it_s / 3600, it_s=it_s))

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
                         "xtick.color": "#5c5c57", "ytick.color": "#5c5c57", "text.color": "#21211f",
                         "axes.labelcolor": "#21211f"})
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.width * 0.38), sharey=True)
    for ax, key, xlabel in ((axes[0], "steps", "Training steps"), (axes[1], "hours", "Training time (GPU hours)")):
        for i, d in enumerate(data):
            x = d[key] / 1000 if key == "steps" else d[key]
            ax.plot(x, d["loss"], color=COLORS[i], lw=1.8, label=d["label"], zorder=3)
            if key == "hours":
                ax.annotate(f"{d['loss'][-1]:.3f}", (x[-1], d["loss"][-1]), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color="#3d3d3a", annotation_clip=False)
        ax.set_xlabel(xlabel)
        ax.grid(True, color="#e4e4e0", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.6)
        if key == "steps":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}K"))
    axes[0].set_ylabel(f"Training loss ({args.window // 1000}K-step mean)")
    axes[0].set_title("(a) Loss vs. training steps", loc="left")
    axes[1].set_title("(b) Loss vs. training time", loc="left")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.5)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.name)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "train_steps", "gpu_hours", "loss_trailing_mean"])
        for d in data:
            for s, h, l in zip(d["steps"][::10], d["hours"][::10], d["loss"][::10]):
                w.writerow([d["label"], int(s), round(float(h), 3), round(float(l), 5)])
    print(f"saved {base}.pdf / .png / .csv   (it/s used: " + ", ".join(f"{d['label']} {d['it_s']:.2f}" for d in data) + ")")

    ref = data[0]
    for d in data[1:]:
        print(f"\n{d['label']} vs {ref['label']}")
        print(f"{'':>10} {'step':>8} {'hours':>6} {'loss':>7} | {'base loss, same steps':>21} | {'base loss, same hours':>21}")
        for st in (50_000, 100_000, 200_000, 300_000, 400_000):
            if st > d["steps"][-1]:
                continue
            l = np.interp(st, d["steps"], d["loss"]); h = st / d["it_s"] / 3600
            bs = np.interp(st, ref["steps"], ref["loss"])
            bh = np.interp(h, ref["hours"], ref["loss"]) if h <= ref["hours"][-1] else float("nan")
            print(f"{'':>10} {st // 1000:>7}K {h:>6.2f} {l:>7.4f} | {bs:>21.4f} | {bh:>21.4f}")
        k = np.argmax(ref["loss"] <= d["loss"][-1])
        if ref["loss"][k] <= d["loss"][-1]:
            print(f"{ref['label']} reaches the final loss of {d['label']} ({d['loss'][-1]:.4f}) at step {ref['steps'][k]:,} "
                  f"= {ref['hours'][k]:.1f} h; {d['label']} needs {d['hours'][-1]:.1f} h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", help='"Label=logs/train/<job>.err"; first one is the baseline.')
    p.add_argument("--speed", action="append", type=float, help="Training it/s per run (default: median of its log).")
    p.add_argument("--window", type=int, default=10_000, help="Trailing-mean window in steps.")
    p.add_argument("--start", type=int, default=10_000, help="First step drawn (the first steps are far above the rest).")
    p.add_argument("--width", type=float, default=7.0)
    p.add_argument("--out-dir", default=os.path.join(REPO, "results", "figures"))
    p.add_argument("--name", default="loss")
    main(p.parse_args())
