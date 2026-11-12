"""
Two summary figures for the paper, from numbers already measured (edit the dicts when new runs land).

  efficiency_gains : (a) method / baseline gains for S/2 and B/2 (train speed, sampling speed, peak memory),
                     (b) forward FLOPs of the method relative to the baseline for S, B, L, XL.
  cost_quality     : FID-50K against training GPU hours: each run at 400K steps, plus the baseline at the
                     method's training time (interpolated between its own scored checkpoints).

    python plot_efficiency.py            # -> results/figures/efficiency_gains.pdf/.png, cost_quality.pdf/.png

Sources: bench_cost.py on gpu03 (A100-PCIe 40GB, batch 256, bf16, compile) = job 79116; FID-50K from
eval_fid.py (SDE Euler 250 steps, no CFG); training it/s = median of each run's own log on gpu02/gpu03.
"""
import os

import numpy as np

# bench_cost.py, job 79116 (gpu03)
BENCH = {
    "SiT-S/2": dict(train=(8.14, 14.40), sample=(8.96, 18.20), mem=(12.2, 6.5)),
    "SiT-B/2": dict(train=(3.13, 5.40), sample=(3.69, 7.10), mem=(25.4, 14.3)),
}
# forward GFLOPs per image (baseline, method); S and B trained, L and XL computed only
GFLOPS = {"S/2": (6.05, 3.61), "B/2": (23.00, 14.01), "L/2": (80.69, 41.44), "XL/2": (118.62, 54.36)}
TRAINED = {"S/2", "B/2"}

# FID-50K per checkpoint (steps -> FID) and training it/s from the logs
RUNS = {
    "SiT-S/2": dict(it_s=(7.75, 13.29),
                    base={50e3: 108.11, 100e3: 80.52, 150e3: 69.81, 200e3: 63.70, 250e3: 59.63, 300e3: 56.47,
                          350e3: 53.81, 400e3: 51.91},
                    method={400e3: 52.28}),
    "SiT-B/2": dict(it_s=(2.98, 5.13),
                    base={400e3: 26.90},          # add 200e3 and 250e3 when their FID jobs finish
                    method={400e3: 27.54}),
}

BLUE = "#2a78d6"     # accent: the method
GREY = "#8f8f89"     # baseline
INK, INK2, GRID, AXIS = "#21211f", "#5c5c57", "#e4e4e0", "#8a8a85"


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.labelcolor": INK})


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def efficiency_gains(plt, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    # (a) gains, all as "x better" so every bar reads the same way: speed = method/base, memory = base/method
    metrics = [("Training\nthroughput", "train", False), ("Sampling\nthroughput", "sample", False),
               ("Peak memory\n(reduction)", "mem", True)]
    sizes = list(BENCH)
    shades = [0.55, 1.0]                      # S/2 lighter, B/2 darker: one hue, ordered by size
    w = 0.34
    for j, size in enumerate(sizes):
        vals = []
        for _, key, inverse in metrics:
            base, meth = BENCH[size][key]
            vals.append(base / meth if inverse else meth / base)
        xs = np.arange(len(metrics)) + (j - 0.5) * w
        a.bar(xs, vals, width=w * 0.92, color=BLUE, alpha=shades[j], label=size, zorder=3)
        for x, v in zip(xs, vals):
            a.text(x, v + 0.04, f"{v:.2f}×", ha="center", va="bottom", fontsize=7.5, color=INK)
    a.axhline(1.0, color=AXIS, lw=0.9, ls=(0, (3, 2)), zorder=2)
    a.set_xticks(np.arange(len(metrics)), [m[0] for m in metrics])
    a.set_ylim(0, 2.6)
    a.set_ylabel("Gain over baseline (×; baseline = 1)")
    a.set_title("(a) Measured gains, same A100", loc="left")
    a.legend(frameon=False, loc="upper left", ncol=2)
    a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    clean(a)

    # (b) FLOPs ratio by model size; hollow = not trained
    names = list(GFLOPS)
    ratio = [GFLOPS[n][1] / GFLOPS[n][0] for n in names]
    x = np.arange(len(names))
    b.plot(x, ratio, color=BLUE, lw=1.6, zorder=3)
    for xi, n, r in zip(x, names, ratio):
        trained = n in TRAINED
        b.plot(xi, r, "o", ms=6.5, color=BLUE, mfc=BLUE if trained else "white", mew=1.5, zorder=4)
        b.text(xi, r + 0.045, f"{r:.2f}×", ha="center", va="bottom", fontsize=7.5, color=INK)
    b.axhline(1.0, color=AXIS, lw=0.9, ls=(0, (3, 2)))
    b.text(len(names) - 1, 1.02, "baseline", ha="right", va="bottom", fontsize=7, color=INK2)
    b.set_xticks(x, [f"SiT-{n}\n{GFLOPS[n][0]:.1f}→{GFLOPS[n][1]:.1f}" for n in names])
    b.set_ylim(0, 1.15)
    b.set_ylabel("Forward FLOPs, method / baseline")
    b.set_xlabel("GFLOPs per image, baseline → method", fontsize=7.5, color=INK2)
    b.set_title("(b) FLOPs saved grows with model size", loc="left")
    b.plot([], [], "o", color=BLUE, mfc=BLUE, label="trained")
    b.plot([], [], "o", color=BLUE, mfc="white", mew=1.5, label="computed only")
    b.legend(frameon=False, loc="lower left", fontsize=7.5)
    b.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    clean(b)
    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"efficiency_gains.{ext}"), dpi=200, bbox_inches="tight")


def cost_quality(plt, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    markers = {"SiT-S/2": "o", "SiT-B/2": "s"}
    notes = []
    xmax = max(400e3 / r["it_s"][0] / 3600 for r in RUNS.values())
    for size, r in RUNS.items():
        itb, itm = r["it_s"]
        bs = np.array(sorted(r["base"])); bf = np.array([r["base"][s] for s in bs]); bh = bs / itb / 3600
        mh = 400e3 / itm / 3600; mf = r["method"][400e3]
        ax.plot(bh, bf, color=GREY, lw=1.2, zorder=2)                        # baseline's own curve
        ax.plot(bh[-1], bf[-1], markers[size], ms=7, color=GREY, zorder=4)
        ax.plot(mh, mf, markers[size], ms=7, color=BLUE, zorder=5)
        far_right = bh[-1] > 0.8 * xmax   # near the right edge: label above-left instead of to the right
        ax.annotate(f"{size} baseline  {bf[-1]:.1f}", (bh[-1], bf[-1]),
                    xytext=(-4, 10) if far_right else (8, 0), textcoords="offset points", fontsize=7.5, color=INK2,
                    ha="right" if far_right else "left", va="bottom" if far_right else "center")
        ax.annotate(f"{size} + ours  {mf:.1f}", (mh, mf), xytext=(0, -10), textcoords="offset points",
                    fontsize=7.5, color=BLUE, ha="center", va="top")
        if bh[0] <= mh <= bh[-1]:   # baseline at the method's training time, from its own scored checkpoints
            eq = float(np.interp(mh, bh, bf))
            ax.plot(mh, eq, markers[size], ms=7, mfc="white", color=GREY, mew=1.5, zorder=5)
            ax.annotate("", xy=(mh, mf + 0.6), xytext=(mh, eq - 0.6),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1))
            ax.annotate(f"same {mh:.1f} GPU h: {eq:.1f} → {mf:.1f}", (mh, eq), xytext=(9, 12),
                        textcoords="offset points", fontsize=7, color=INK, va="bottom", ha="left")
        else:
            notes.append(f"{size}: baseline at {mh:.1f} GPU h pending (FID at 200K/250K)")
    ax.set_xlabel("Training time to 400K steps (GPU hours, A100-PCIe)")
    ax.set_ylabel("FID-50K")
    ax.set_title("Same quality for less compute; better at equal compute", loc="left")
    ax.plot([], [], "o", color=GREY, label="baseline, 400K steps")
    ax.plot([], [], "o", color=BLUE, label="+ token compression, 400K steps")
    ax.plot([], [], "o", color=GREY, mfc="white", mew=1.5, label="baseline at the method's time")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    clean(ax)
    if notes:
        fig.text(0.01, -0.02, "; ".join(notes), fontsize=6.5, color=INK2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"cost_quality.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    out = "results/figures"
    os.makedirs(out, exist_ok=True)
    efficiency_gains(plt, out)
    cost_quality(plt, out)
    print(f"saved {out}/efficiency_gains.* and {out}/cost_quality.*")
