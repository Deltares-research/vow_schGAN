"""Summary figures for the borehole-vs-CPT pair evaluation (all pairs pooled)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # add src/ to sys.path

from modules.borehole_ic.pair_evaluation import ROBERTSON_THRESHOLDS  # noqa: E402
from modules.borehole_ic.triangles import CLASS_COLORS  # noqa: E402

_CATEGORY_COLORS = {"exact": "tab:blue", "guessed": "tab:orange", "interpolated": "tab:purple", "transition": "tab:gray"}
_FAMILY_ORDER = ["Z", "L", "K", "V", "G"]


def _class_sort_key(code: str):
    return (_FAMILY_ORDER.index(code[0]) if code and code[0] in _FAMILY_ORDER else 99, code)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _zone_grid(ax):
    for t in ROBERTSON_THRESHOLDS:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.7)
        ax.axhline(t, color="gray", linestyle=":", linewidth=0.7)


def plot_scatter(pooled: pd.DataFrame, layer_table: pd.DataFrame, path: Path):
    """Predicted vs measured Ic: every compared depth sample (hexbin) and per-layer means."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    lo = np.nanmin([pooled["ic_median"].min(), pooled["reference_ic"].min()]) - 0.1
    hi = np.nanmax([pooled["ic_median"].max(), pooled["reference_ic"].max()]) + 0.1

    ax = axes[0]
    hb = ax.hexbin(pooled["reference_ic"], pooled["ic_median"], gridsize=60, bins="log", cmap="viridis", mincnt=1,
                   extent=(lo, hi, lo, hi))
    fig.colorbar(hb, ax=ax, label="number of depth samples (log)")
    ax.set_title(f"All compared depth samples (n = {len(pooled)})")

    ax = axes[1]
    for row in layer_table.itertuples():
        ax.errorbar(row.measured_ic_mean, row.synthetic_ic_mean, xerr=row.measured_ic_std, fmt="none",
                    ecolor="gray", elinewidth=0.6, alpha=0.6, zorder=1)
    colors = [CLASS_COLORS.get(c, "#dddddd") for c in layer_table["class_code"]]
    markers = np.where(layer_table["is_guess"], "s", "o")
    for marker in ("o", "s"):
        sel = markers == marker
        ax.scatter(layer_table.loc[sel, "measured_ic_mean"], layer_table.loc[sel, "synthetic_ic_mean"],
                   c=np.array(colors, dtype=object)[sel], edgecolors="black", linewidths=0.5, s=40, zorder=2,
                   marker=marker)
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="lightgray", markeredgecolor="black", label="exact class"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor="lightgray", markeredgecolor="black",
               label="guessed class (*)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left", title="colour = borehole class", title_fontsize=8)
    ax.set_title(f"Per borehole layer: mean Ic (n = {len(layer_table)} layers; bars = measured std)")

    for ax in axes:
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        _zone_grid(ax)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Measured CPT Ic (-)")
        ax.set_ylabel("Predicted Ic, ensemble median (-)")
    fig.suptitle("Predicted vs measured Ic (dotted = Robertson zone limits, line = 1:1)")
    _save(fig, path)


def plot_error_by_class(pooled: pd.DataFrame, path: Path):
    """Box plot of the error per source class (transition samples excluded)."""
    in_layer = pooled[~pooled["is_transition"]]
    codes = sorted(in_layer["source_class_code"].dropna().unique(), key=_class_sort_key)
    if not codes:
        return
    data = [in_layer.loc[in_layer["source_class_code"] == c, "error"].to_numpy() for c in codes]
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(codes) + 3), 6))
    box = ax.boxplot(data, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
    for patch, code in zip(box["boxes"], codes):
        patch.set_facecolor(CLASS_COLORS.get(code, "#dddddd"))
    ax.axhline(0, color="black", linewidth=0.8)
    tick_labels = [
        f"{code}\nn={len(values)}\n{in_layer.loc[in_layer['source_class_code'] == code, 'pair_id'].nunique()} pair(s)"
        for code, values in zip(codes, data)
    ]
    ax.set_xticks(range(1, len(codes) + 1))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("Error: predicted - measured Ic (-)")
    ax.set_xlabel("Borehole class (* = generic guess)")
    ax.set_title("Ic error per borehole class (boxes = 25-75%, whiskers = 1.5 IQR, outliers hidden)")
    _save(fig, path)


def plot_error_per_pair(pair_metrics: pd.DataFrame, path: Path):
    """Bias and MAE per pair (sorted by MAE) and MAE against CPT-borehole distance."""
    ok = pair_metrics[pair_metrics["status"] == "ok"].sort_values("mae")
    if not len(ok):
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(12, 0.35 * len(ok) + 6), 6), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    x = np.arange(len(ok))
    ax.bar(x, ok["bias"], color=np.where(ok["bias"] >= 0, "tab:red", "tab:blue"), alpha=0.7, label="bias (mean error)")
    ax.scatter(x, ok["mae"], color="black", marker="D", s=18, zorder=3, label="MAE")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ok["pair_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Ic error (-)")
    ax.set_title("Per pair: bias (bars) and mean absolute error (diamonds), sorted by MAE")
    ax.legend(fontsize=8)

    ax = axes[1]
    sc = ax.scatter(ok["distance_m"], ok["mae"], c=ok["fraction_inside_band"], cmap="RdYlGn", vmin=0, vmax=1,
                    edgecolors="black", linewidths=0.4)
    fig.colorbar(sc, ax=ax, label="fraction of measured Ic inside predicted band")
    ax.set_xlabel("CPT-borehole distance (m)")
    ax.set_ylabel("MAE (-)")
    ax.set_title("MAE vs distance")
    _save(fig, path)


def plot_zone_confusion(confusion: pd.DataFrame, path: Path):
    """Robertson zone of the measured CPT (rows) vs the prediction (columns), row-normalised."""
    counts = confusion.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="fraction of measured-zone samples")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            if counts[i, j]:
                ax.text(j, i, f"{frac[i, j]:.0%}\n({int(counts[i, j])})", ha="center", va="center", fontsize=7,
                        color="white" if frac[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_xticklabels(confusion.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(confusion.index)))
    ax.set_yticklabels(confusion.index, fontsize=8)
    ax.set_xlabel("Predicted zone (ensemble median)")
    ax.set_ylabel("Measured zone (CPT)")
    agreement = np.trace(counts) / counts.sum() if counts.sum() else np.nan
    ax.set_title(f"Robertson zone agreement: {agreement:.0%} of compared samples")
    _save(fig, path)


def plot_error_histogram(pooled: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(-2.0, 2.0, 81)
    for category, color in _CATEGORY_COLORS.items():
        err = pooled.loc[pooled["sample_category"] == category, "error"].clip(-2.0, 2.0)
        if len(err):
            ax.hist(err, bins=bins, alpha=0.5, color=color,
                    label=f"{category} (n={len(err)}, bias {err.mean():+.2f}, MAE {err.abs().mean():.2f})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Error: predicted - measured Ic (-), clipped to +/-2")
    ax.set_ylabel("Number of depth samples")
    ax.set_title("Distribution of Ic errors by sample type")
    ax.legend(fontsize=8)
    _save(fig, path)
