#!/usr/bin/env python3
"""
Docstring for plotting.plot_pca_specturm
python3 plot_pca_specturm.py \
  --json prove_existence_meta-llama_Llama-2-7b-chat-hf_fp32_layer24_n128_new256_maxlen512_states20000_tau0.001_msharedall_perm1_scr0.json \
         prove_existence_meta-llama_Llama-2-7b-chat-hf_fp32_layer30_n128_new256_maxlen512_states20000_tau0.001_msharedall_perm1_scr0.json \
  --n 200 \
  --logy
"""


from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _parse_n_or_range(s: str) -> tuple[int, int]:
    """
    Parse --n either as:
    - "200" -> (0, 199)
    - "100-120" -> (100, 120) inclusive
    """
    s = s.strip()
    if "-" in s:
        a_str, b_str = s.split("-", 1)
        return int(a_str), int(b_str)
    n = int(s)
    return 0, n - 1




def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, nargs="+", help="One or more results JSON paths.")
    p.add_argument(
        "--n",
        type=str,
        default="200",
        help="Number of PCs from the front (e.g. '200') or a range like '100-120' (inclusive).",
    )
    p.add_argument("--logy", action="store_true", help="Use log-scale y-axis.")
    p.add_argument(
        "--out",
        default=None,
        help="Output PDF path (default: spectrum-layers<L1>-<L2>-....pdf).",
    )
    args = p.parse_args()

    ds = [_load_json(pth) for pth in args.json]

    # ---- slice spectrum ----
    evrs: list[np.ndarray] = []
    layers: list[int | None] = []
    shared_sets: list[set[int]] = []

    for d in ds:
        pca = d.get("pca_spectrum", {})
        evr = np.asarray(pca.get("explained_variance_ratio", None), dtype=np.float64)
        evrs.append(evr)

        layers.append(d.get("config", {}).get("layer", None))

        shared_indices = d.get("observed", {}).get("shared_indices", [])
        shared_sets.append(set(int(x) for x in shared_indices))

    start, end_inclusive = _parse_n_or_range(args.n)
    start = max(0, start)
    min_len = min(len(evr) for evr in evrs)
    end_inclusive = min(end_inclusive, min_len - 1)
    if end_inclusive < start:
        raise ValueError(f"Empty PC range: start={start}, end_inclusive={end_inclusive}.")
    end_exclusive = end_inclusive + 1

    xs = np.arange(start, end_exclusive)
    yss = [evr[start:end_exclusive] for evr in evrs]

    # --- palette (keep consistent "decodeshare" style) ---
    # Provided: #bdfcf6, #4b64a3, #19d8f7
    C_ACCENT_LIGHT = "#bdfcf6"
    C_PRIMARY_DARK = "#4b64a3"
    C_ACCENT_BRIGHT = "#19d8f7"
    # Distinct cumulative-curve color (still same cool palette, but not equal to shared bars).
    C_CUMULATIVE = "#2fbfbd"

    # Avoid 10^-2 style ticks; use compact decimals.
    def _plain_tick(y: float, _pos: int) -> str:
        if y == 0:
            return "0"
        # Hide 0.001 on axis (we'll annotate inside the plot if visible).
        if np.isclose(y, 1e-3, rtol=0.0, atol=1e-12):
            return ""
        if 1e-4 <= abs(y) < 1:
            return f"{y:.4f}".rstrip("0").rstrip(".")
        return f"{y:.3g}"

    plt.rcParams.update({
        "font.size": 29,
        "font.weight": "bold",
        "axes.titlesize": 29,
        "axes.labelsize": 29,
        "axes.labelweight": "bold",
        "xtick.labelsize": 29,
        "ytick.labelsize": 29,
    })
    # Less "flat": make each subplot taller and overall wider.
    fig_w = 10.5
    fig_h = 6.8 if len(yss) == 2 else (4.6 * len(yss))
    fig, axes = plt.subplots(
        nrows=len(yss),
        ncols=1,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(yss) == 1:
        axes = np.asarray([axes])

    for ax, ys, layer, shared_set, evr_full in zip(axes, yss, layers, shared_sets, evrs):
        # Color: shared = bright accent, non-shared = dark primary.
        colors = [C_ACCENT_BRIGHT if int(i) in shared_set else C_PRIMARY_DARK for i in xs]
        # Subtle shared-block highlight to add structure.
        in_range_shared = sorted(i for i in shared_set if start <= i <= end_inclusive)
        if in_range_shared:
            run_start = in_range_shared[0]
            prev = run_start
            for i in in_range_shared[1:] + [None]:
                if i is not None and i == prev + 1:
                    prev = i
                    continue
                ax.axvspan(run_start - 0.5, prev + 0.5, color=C_ACCENT_LIGHT, alpha=0.10, lw=0, zorder=0)
                if i is None:
                    break
                run_start = prev = i

        ax.bar(xs, ys, width=0.98, color=colors, linewidth=0, zorder=2)

        # Cumulative EVR on a clean secondary axis (explains how much variance we've covered).
        ax2 = ax.twinx()
        cum_full = np.cumsum(evr_full)
        cum_seg = cum_full[start:end_exclusive]
        ax2.plot(xs, cum_seg, color=C_CUMULATIVE, linewidth=2.6, alpha=0.98)
        ax2.set_ylim(0.0, 1.0)
        # Remove all y-axis numbers (also on the cumulative axis).
        ax2.set_yticks([])
        ax2.tick_params(axis="y", which="both", length=0, labelright=False, colors=C_PRIMARY_DARK)
        # Avoid extra label text (can overlap with other annotations).
        ax2.set_ylabel("")
        for sp in ax2.spines.values():
            sp.set_visible(False)
        # Label end-of-range cumulative value.
        if len(xs) > 0:
            # Put the number on the curve end-point (inside the plot).
            ax2.annotate(
                f"{float(cum_seg[-1]) * 100:.1f}%",
                xy=(xs[-1], float(cum_seg[-1])),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=26,
                fontweight="bold",
                color=C_PRIMARY_DARK,
                alpha=0.9,
                clip_on=True,
            )

        # Clean frame: keep y-axis spine, no grid.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, length=0)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(C_PRIMARY_DARK)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # No split/grid lines; show readable y tick values.
        ax.grid(False)
        # Remove all y-axis numbers on the EVR axis.
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, length=0, colors=C_PRIMARY_DARK)

        if args.logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.offsetText.set_visible(False)
            # Put 0.001 into the plot (instead of the axis) if it's within view.
            ylo, yhi = ax.get_ylim()
            if ylo <= 1e-3 <= yhi and len(xs) > 0:
                ax.annotate(
                    "0.001",
                    xy=(xs[0], 1e-3),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=21,
                    fontweight="bold",
                    color=C_PRIMARY_DARK,
                    alpha=0.9,
                )
        else:
            ax.yaxis.set_major_formatter(NullFormatter())

        # In-plot layer tag (so we can tell subplots apart without extra axes labels).
        if layer is not None:
            ax.text(
                0.01,
                0.86,
                f"Layer {layer}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=25,
                fontweight="bold",
                color=C_PRIMARY_DARK,
            )

    if args.logy:
        all_y = np.concatenate([ys for ys in yss], axis=0)
        ymin_pos = all_y[all_y > 0].min() if np.any(all_y > 0) else 1e-12
        ymax = float(all_y.max()) if all_y.size else 1.0
        axes[0].set_ylim(bottom=ymin_pos * 0.9, top=ymax * 3.0)
    else:
        all_y = np.concatenate([ys for ys in yss], axis=0)
        ymax = float(all_y.max()) if all_y.size else 1.0
        axes[0].set_ylim(bottom=0.0, top=ymax * 1.25)



    # Shared block bracket

    # One shared set of axis labels/ticks for the full figure.
    fig.supylabel("Explained variance", fontsize=29, fontweight="bold")
    # Show x ticks only on the bottom axis (single shared x-axis coordinate set).
    tick_idx = np.unique(np.clip(np.linspace(start, end_inclusive, 5).round().astype(int), start, end_inclusive))
    axes[-1].set_xticks(tick_idx)
    axes[-1].tick_params(axis="x", which="both", bottom=True, labelbottom=True, length=0, colors=C_PRIMARY_DARK)
    axes[-1].set_xlabel("PC index", fontsize=25, fontweight="bold", color=C_PRIMARY_DARK, labelpad=10)
    for t in axes[-1].get_xticklabels():
        t.set_fontweight("bold")

    # Compact legend so readers can decode colors/line.
    legend_items = [
        Patch(facecolor=C_PRIMARY_DARK, edgecolor="none", label="Non-shared"),
        Patch(facecolor=C_ACCENT_BRIGHT, edgecolor="none", label="Shared"),
        Line2D([0], [0], color=C_CUMULATIVE, lw=2.6, label="Cumulative EVR"),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=23,
        prop={"weight": "bold", "size": 23},
        bbox_to_anchor=(0.5, 1.02),
    )

    layer_tags = [str(l) if l is not None else "unknown" for l in layers]
    out_path = args.out or f"spectrum-layers{'-'.join(layer_tags)}.pdf"


    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
