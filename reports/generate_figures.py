"""
Generate all summary figures for the comprehensive project report.
Run from the repo root:  python reports/generate_figures.py
Dependencies: numpy, matplotlib (both already available)
"""

import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_FFT    = "#4C72B0"   # blue
C_MOMENT = "#55A868"   # green
C_TS2VEC = "#C44E52"   # red
C_SUMM   = "#8172B2"   # purple
COLOURS  = [C_FFT, C_MOMENT, C_TS2VEC, C_SUMM]

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1  RQ1 — RBF probe balanced accuracy (clean dataset, 4 factors)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_rq1_probes():
    factors = ["Mode", "Spectral", "Coupling", "Transition"]
    methods = ["FFT", "Summary", "MOMENT", "TS2Vec e80", "TS2Vec e120"]
    colours = [C_FFT, C_SUMM, C_MOMENT, C_TS2VEC, "#E07B39"]

    data = {
        "FFT":          [0.439, 0.696, 0.673, None],
        "Summary":      [0.431, 0.653, 0.634, None],
        "MOMENT":       [0.466, 0.756, 0.633, 0.576],
        "TS2Vec e80":   [0.398, 0.694, 0.625, 0.527],
        "TS2Vec e120":  [0.514, 0.767, 0.713, 0.527],
    }
    err = {
        "MOMENT": [0.012, 0.018, 0.042, 0.042],
    }

    fig, ax = plt.subplots(figsize=(11, 5))
    n_groups, n_methods = len(factors), len(methods)
    width = 0.15
    x = np.arange(n_groups)

    for i, (m, col) in enumerate(zip(methods, colours)):
        vals = data[m]
        yerr_vals = err.get(m, [None]*4)
        # plot bars; skip None values
        for j, (v, e) in enumerate(zip(vals, yerr_vals)):
            if v is None:
                continue
            bar_x = x[j] + (i - n_methods/2 + 0.5) * width
            ax.bar(bar_x, v, width*0.9, color=col, alpha=0.88,
                   yerr=e, capsize=3, error_kw={"elinewidth":1.2},
                   label=m if j == 0 else "_nolegend_")

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=12)
    ax.set_ylabel("RBF Probe Balanced Accuracy", fontsize=11)
    ax.set_title("RQ1 — Latent Factor Recovery (clean dataset, RBF probes)", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 0.85)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    ax.annotate("Baselines have no\nTransition probe", xy=(3, 0.31), fontsize=8, color="gray", ha="center")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_rq1_probes.png", dpi=150)
    plt.close()
    print("fig1 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  RQ1 — clean vs noisy average accuracy + clean/noisy degradation
# ─────────────────────────────────────────────────────────────────────────────
def fig2_rq1_clean_noisy():
    methods = ["FFT", "Summary", "MOMENT", "TS2Vec e80", "TS2Vec e120"]
    cols    = [C_FFT, C_SUMM, C_MOMENT, C_TS2VEC, "#E07B39"]
    clean   = [0.603, 0.573, 0.618, 0.572, 0.665]
    noisy   = [0.576, 0.579, 0.590, 0.575, 0.607]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_c = ax.bar(x - w/2, clean, w, color=cols, alpha=0.9, label="Clean")
    bars_n = ax.bar(x + w/2, noisy, w, color=cols, alpha=0.45, hatch="//", label="Noisy")

    for xc, vc, vn in zip(x, clean, noisy):
        drop = vc - vn
        ax.annotate(f"−{drop:.3f}", xy=(xc, max(vc,vn)+0.002),
                    ha="center", fontsize=8, color="#555")

    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Avg Balanced Accuracy (mode+spectral+coupling)", fontsize=10)
    ax.set_title("RQ1 — Clean vs Noisy: All Methods", fontsize=13, fontweight="bold")
    ax.set_ylim(0.45, 0.72)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.9, label="Clean"),
        Patch(facecolor="gray", alpha=0.45, hatch="//", label="Noisy"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_rq1_clean_noisy.png", dpi=150)
    plt.close()
    print("fig2 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  RQ1 — Clustering ARI
# ─────────────────────────────────────────────────────────────────────────────
def fig3_rq1_clustering():
    methods = ["MOMENT", "TS2Vec e80", "TS2Vec e120"]
    cols    = [C_MOMENT, C_TS2VEC, "#E07B39"]
    data = {
        "MOMENT":       [0.151, 0.180, 0.058],
        "TS2Vec e80":   [0.153, 0.207, 0.006],
        "TS2Vec e120":  [0.157, 0.194, 0.004],
    }
    factors = ["Mode", "Spectral", "Coupling"]

    x = np.arange(len(factors))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, (m, col) in enumerate(zip(methods, cols)):
        ax.bar(x + (i-1)*width, data[m], width*0.9, color=col, alpha=0.88, label=m)

    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(factors, fontsize=12)
    ax.set_ylabel("Clustering ARI", fontsize=11)
    ax.set_title("RQ1 — Global Clustering Quality (ARI, clean dataset)", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 0.26)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    ax.text(0.98, 0.95, "ARI < 0.21 across the board\n— global structure absent",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color="#c00", bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#c00", alpha=0.8))
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_rq1_clustering.png", dpi=150)
    plt.close()
    print("fig3 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4  RQ2 — Winner heatmap (3 problems × 5 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig4_rq2_winner_heatmap():
    # winner matrix: 0=fft, 1=moment, 2=ts2vec
    # metrics: MSI↑  LoopDTW↓  TransSharp↓  PCACom↓  CentStab↓
    # p1: moment, moment, fft/ts2vec(tie→fft), moment, moment
    # p2: moment, moment, moment, fft, moment
    # p3: fft,    moment, moment, fft, moment
    winners = np.array([
        [1, 1, 0, 1, 1],   # P1
        [1, 1, 1, 0, 1],   # P2
        [0, 1, 1, 0, 1],   # P3
    ])

    problems = ["P1\nSimple 1D", "P2\nMulti-channel", "P3\nHard+Noisy"]
    metrics  = ["MSI\n(↑ better)", "Loop DTW\n(↓ better)", "Trans. Sharp\n(↓ better)",
                "PCA Compact\n(↓ better)", "Centroid Stab\n(↓ better)"]

    # actual values for annotation
    vals = {
        "fft":    [[11.1, 66.6, 3.0,  0.50, 0.289],
                   [22.5, 146.1, 3.3, 0.26, 0.278],
                   [20.2, 167.4, 4.0, 0.33, 0.364]],
        "moment": [[71.1, 2.76, 5.7,  0.23, 0.002],
                   [27.2, 1.43, 1.67, 1.63, 0.001],
                   [14.3, 1.64, 1.33, 7.97, 0.001]],
        "ts2vec": [[20.0, 44.6, 3.0,  0.37, 0.085],
                   [4.4, 108.2, 3.67, 48.4, 0.509],
                   [4.6, 107.8, 4.0,  55.9, 0.529]],
    }
    model_list = ["fft", "moment", "ts2vec"]

    cmap = matplotlib.colors.ListedColormap([C_FFT, C_MOMENT, C_TS2VEC])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(winners, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(problems)))
    ax.set_yticklabels(problems, fontsize=11)
    ax.set_title("RQ2 — Model Winner per Problem × Metric", fontsize=13, fontweight="bold")

    # annotate cells with winning model name + its value
    for i in range(3):
        for j in range(5):
            w = winners[i, j]
            m = model_list[w]
            v = vals[m][i][j]
            fmt = f"{v:.1f}" if v >= 1 else f"{v:.3f}"
            ax.text(j, i, f"{m.upper()}\n{fmt}", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white")

    legend_patches = [
        mpatches.Patch(color=C_FFT,    label="FFT wins"),
        mpatches.Patch(color=C_MOMENT, label="MOMENT wins"),
        mpatches.Patch(color=C_TS2VEC, label="TS2Vec wins"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", bbox_to_anchor=(1.0, -0.28),
              ncol=3, fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_rq2_winner_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig4 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5  RQ2 — MSI and Loop Consistency side-by-side bars
# ─────────────────────────────────────────────────────────────────────────────
def fig5_rq2_msi_loop():
    problems = ["P1\nSimple 1D", "P2\nMulti-ch", "P3\nHard Noisy"]
    msi  = {"FFT":    [11.1, 22.5, 20.2],
            "MOMENT": [71.1, 27.2, 14.3],
            "TS2Vec": [20.0,  4.4,  4.6]}
    loop = {"FFT":    [66.6, 146.1, 167.4],
            "MOMENT": [ 2.8,   1.4,   1.6],
            "TS2Vec": [44.6, 108.2, 107.8]}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cols   = [C_FFT, C_MOMENT, C_TS2VEC]
    models = ["FFT", "MOMENT", "TS2Vec"]
    x = np.arange(3)
    w = 0.25

    for ax, metric, title, ylabel, higher_better in zip(
            axes,
            [msi, loop],
            ["Mode Separability Index (MSI, ↑ better)",
             "Loop Consistency — DTW (↓ better)"],
            ["MSI", "DTW distance"],
            [True, False]):

        for i, (m, col) in enumerate(zip(models, cols)):
            bars = ax.bar(x + (i-1)*w, metric[m], w*0.9, color=col, alpha=0.88, label=m)

        ax.set_xticks(x)
        ax.set_xticklabels(problems, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

        if not higher_better:
            ax.set_yscale("log")
            ax.set_ylabel(ylabel + " (log scale)", fontsize=11)

    fig.suptitle("RQ2 — Core Embedding Quality Metrics", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_rq2_msi_loop.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig5 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6  RQ2 — Per-model radar chart (averaged over problems)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_rq2_radar():
    # Normalise each metric to [0,1] where 1 = best possible
    # MSI: higher→better; normalise by max across all 9
    # Loop DTW: lower→better; invert then normalise
    # Trans Sharp: lower→better; invert
    # PCA Compact: lower→better; invert
    # Centroid Stab: lower→better; invert

    raw = {
        # [p1, p2, p3] for each metric
        "FFT":    {"msi":   [11.1, 22.5, 20.2],
                   "loop":  [66.6, 146.1, 167.4],
                   "trans": [3.0,  3.33, 4.0],
                   "pca":   [0.50, 0.26, 0.33],
                   "cstab": [0.289, 0.278, 0.364]},
        "MOMENT": {"msi":   [71.1, 27.2, 14.3],
                   "loop":  [2.76, 1.43, 1.64],
                   "trans": [5.67, 1.67, 1.33],
                   "pca":   [0.23, 1.63, 7.97],
                   "cstab": [0.002, 0.001, 0.001]},
        "TS2Vec": {"msi":   [20.0, 4.4, 4.6],
                   "loop":  [44.6, 108.2, 107.8],
                   "trans": [3.0, 3.67, 4.0],
                   "pca":   [0.37, 48.4, 55.9],
                   "cstab": [0.085, 0.509, 0.529]},
    }

    # compute per-model averages across problems
    avgs = {}
    for model, d in raw.items():
        avgs[model] = {k: np.mean(v) for k, v in d.items()}

    # collect all values per metric for normalisation
    all_vals = {k: [] for k in ["msi","loop","trans","pca","cstab"]}
    for d in avgs.values():
        for k, v in d.items():
            all_vals[k].append(v)

    # normalise: for "lower is better" metrics, score = (max-v)/(max-min)
    # for "higher is better": score = (v-min)/(max-min)
    higher_is_better = {"msi": True, "loop": False, "trans": False, "pca": False, "cstab": False}
    scores = {}
    for model, d in avgs.items():
        scores[model] = {}
        for k, v in d.items():
            mn, mx = min(all_vals[k]), max(all_vals[k])
            if mx == mn:
                s = 1.0
            elif higher_is_better[k]:
                s = (v - mn) / (mx - mn)
            else:
                s = (mx - v) / (mx - mn)
            scores[model][k] = s

    labels = ["MSI\n(sep.)", "Loop DTW\n(consist.)", "Trans.\nSharpness",
              "PCA\nCompact.", "Centroid\nStability"]
    keys = ["msi", "loop", "trans", "pca", "cstab"]
    N = len(keys)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    model_colours = {"FFT": C_FFT, "MOMENT": C_MOMENT, "TS2Vec": C_TS2VEC}

    for model, col in model_colours.items():
        vals = [scores[model][k] for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", lw=2, color=col, label=model)
        ax.fill(angles, vals, alpha=0.12, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax.set_title("RQ2 — Normalised Average Score per Model\n(outer edge = best)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_rq2_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig6 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7  Cross-experiment comparison: MOMENT and TS2Vec role reversal
# ─────────────────────────────────────────────────────────────────────────────
def fig7_role_reversal():
    """MOMENT beats TS2Vec on RQ2 geometry metrics, but TS2Vec beats MOMENT
    on RQ1 factor recovery probes.  Illustrate the contrast."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RQ1 avg balanced accuracy (3 classification factors, clean)
    labels_rq1 = ["Mode", "Spectral", "Coupling", "Avg"]
    moment_rq1 = [0.466, 0.756, 0.633, 0.618]
    ts2vec_rq1 = [0.514, 0.767, 0.713, 0.665]
    fft_rq1    = [0.439, 0.696, 0.673, 0.603]

    x = np.arange(len(labels_rq1))
    w = 0.25
    ax = axes[0]
    ax.bar(x - w, fft_rq1,    w*0.9, color=C_FFT,    alpha=0.88, label="FFT baseline")
    ax.bar(x,     moment_rq1, w*0.9, color=C_MOMENT,  alpha=0.88, label="MOMENT")
    ax.bar(x + w, ts2vec_rq1, w*0.9, color=C_TS2VEC,  alpha=0.88, label="TS2Vec e120")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_rq1, fontsize=11)
    ax.set_ylim(0.35, 0.82)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("RBF Probe Balanced Accuracy", fontsize=10)
    ax.set_title("RQ1 — Factor Recovery\n(clean dataset)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # Right: RQ2 loop consistency (DTW, averaged over 3 problems) — log scale
    ax2 = axes[1]
    models = ["FFT", "MOMENT", "TS2Vec"]
    cols2  = [C_FFT, C_MOMENT, C_TS2VEC]
    loop_avg = [
        np.mean([66.6, 146.1, 167.4]),
        np.mean([2.76, 1.43, 1.64]),
        np.mean([44.6, 108.2, 107.8]),
    ]
    msi_avg = [
        np.mean([11.1, 22.5, 20.2]),
        np.mean([71.1, 27.2, 14.3]),
        np.mean([20.0, 4.4, 4.6]),
    ]
    x2 = np.arange(3)
    bars = ax2.bar(x2 - 0.2, loop_avg, 0.35, color=cols2, alpha=0.88)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_yscale("log")
    ax2.set_ylabel("Loop Consistency DTW (↓ better, log scale)", fontsize=10, color="navy")
    ax2.set_title("RQ2 — Embedding Geometry\n(Loop Consistency, avg over 3 problems)",
                  fontsize=11, fontweight="bold")
    for bar, v in zip(bars, loop_avg):
        ax2.text(bar.get_x() + bar.get_width()/2, v*1.15, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=9)

    ax2_r = ax2.twinx()
    ax2_r.plot(x2, msi_avg, "D--", color="#333", lw=1.5, ms=7, label="MSI avg (↑)")
    ax2_r.set_ylabel("MSI (↑ better, linear)", fontsize=10, color="#333")
    ax2_r.spines[["top"]].set_visible(False)
    ax2_r.legend(loc="upper right", fontsize=8)
    ax2.spines[["top"]].set_visible(False)

    fig.suptitle("The Role Reversal: TS2Vec beats MOMENT on RQ1 factor probes,\n"
                 "MOMENT dominates on RQ2 embedding geometry",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_role_reversal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig7 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8  RQ2 — Detailed metric breakdown (all 5 metrics, all 9 combos)
# ─────────────────────────────────────────────────────────────────────────────
def fig8_rq2_all_metrics():
    combos = ["P1/FFT","P1/MOM","P1/TS2",
              "P2/FFT","P2/MOM","P2/TS2",
              "P3/FFT","P3/MOM","P3/TS2"]
    cols9 = [C_FFT, C_MOMENT, C_TS2VEC] * 3

    metrics_data = {
        "MSI (↑)":               [11.1, 71.1, 20.0,  22.5, 27.2,  4.4,   20.2, 14.3,  4.6],
        "Loop DTW (↓)":          [66.6,  2.8, 44.6, 146.1,  1.4, 108.2, 167.4,  1.6, 107.8],
        "Trans. Sharp (↓)":      [3.0,   5.7,  3.0,   3.3,  1.7,   3.7,   4.0,  1.3,   4.0],
        "PCA Compact (↓)":       [0.50, 0.23, 0.37,  0.26, 1.63,  48.4,  0.33, 7.97,  55.9],
        "Centroid Stab (↓)":     [0.289, 0.002, 0.085, 0.278, 0.001, 0.509, 0.364, 0.001, 0.529],
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    log_metrics = {"Loop DTW (↓)", "PCA Compact (↓)"}

    for ax, (metric, vals) in zip(axes, metrics_data.items()):
        bars = ax.bar(range(9), vals, color=cols9, alpha=0.85)
        ax.set_xticks(range(9))
        ax.set_xticklabels(combos, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        if metric in log_metrics:
            ax.set_yscale("log")

    # shared legend
    legend_patches = [
        mpatches.Patch(color=C_FFT,    label="FFT"),
        mpatches.Patch(color=C_MOMENT, label="MOMENT"),
        mpatches.Patch(color=C_TS2VEC, label="TS2Vec"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("RQ2 — All 5 Metrics × 9 Combinations (P1–P3 × FFT/MOMENT/TS2Vec)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"{OUT}/fig8_rq2_all_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig8 done")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9  Phase timeline
# ─────────────────────────────────────────────────────────────────────────────
def fig9_timeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    phases = [
        (0, "Phase 1\nContrastive LSTM\non toy logs",
         "No formal eval\nTemporal-neighbour pairs\n→ misaligned", "#d4e6f1"),
        (1, "Phase 2\nTCN Hybrid\n(version2 branch)",
         "Best: coupling +0.010\nover baseline (marginal)\nClustering ARI 0.03–0.13", "#d5f5e3"),
        (2, "Phase 3 / RQ1\nFormal FRS Benchmark\n(experiment1 branch)",
         "TS2Vec e120: avg 66.5%\nFFT baseline: 60.3%\nGap: +6.2pp only\nLoad R²≤0 always", "#fdebd0"),
        (3, "RQ2\nTrace Geometry\nComparison",
         "MOMENT: loop DTW ~1.5\n(50–100× better than FFT)\nFFT beats MOMENT\non noisy MSI", "#e8daef"),
        (4, "RQ3\nLandscape Sweep\n(in progress)",
         "14 scenarios, FFT done\nMOMENT/DCC pending\nEval not yet run", "#fdfefe"),
    ]

    for i, (x, title, result, bg) in enumerate(phases):
        box = mpatches.FancyBboxPatch((x-0.42, -0.8), 0.84, 2.9,
                                       boxstyle="round,pad=0.05",
                                       facecolor=bg, edgecolor="#888", lw=1.2)
        ax.add_patch(box)
        ax.text(x, 1.7, title, ha="center", va="center", fontsize=9.5, fontweight="bold",
                multialignment="center")
        ax.text(x, 0.3, result, ha="center", va="center", fontsize=8,
                multialignment="center", color="#333")

        if i < len(phases) - 1:
            ax.annotate("", xy=(x+0.47, 0.7), xytext=(x+0.43, 0.7),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#555"))

    ax.set_title("Experiment Timeline — Key Results at Each Phase",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("fig9 done")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Writing figures to {OUT}/")
    fig1_rq1_probes()
    fig2_rq1_clean_noisy()
    fig3_rq1_clustering()
    fig4_rq2_winner_heatmap()
    fig5_rq2_msi_loop()
    fig6_rq2_radar()
    fig7_role_reversal()
    fig8_rq2_all_metrics()
    fig9_timeline()
    print("All done.")
