"""Mooncake cache figure in the REFLECT 2AFC bar-plot style.

Layout and palette are taken from REFLECT_2AFC_barplot.py: five colours, Times New
Roman throughout, bar width 0.25 on a 0.3 pitch, black 0.8 edges at alpha 0.95, a
framed five-column legend above the axes, 30 pt bold axis titles and a light grey
horizontal grid. The reference line of the original is dropped because neither
reuse nor occupancy has a meaningful fixed baseline.

Writes figures/fig-cache.{pdf,png,svg} (two panels) and
figures/fig-reuse.{pdf,png,svg} (reuse only). Run inside timeline/h100mc.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_trace import summarize  # noqa: E402

PALETTE = ["#9EC4BE", "#ABD0F1", "#F19685", "#FFB77F", "#FBE8D5"]
ORDER = ["FCFS", "Cache-Aware", "Cache-Aware + Guard", "Session-Aware", "Session-Aware + Cache-Aware + Guard"]
TAG = {"FCFS": "baseline", "Cache-Aware": "kvaware-nh", "Cache-Aware + Guard": "W40",
       "Session-Aware": "retainU", "Session-Aware + Cache-Aware + Guard": "W40-retain"}
LOADS = [("50% Load", "0.5"), ("75% Load", "0.75"), ("100% Load", "1.0")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")

BAR_W, PITCH, GAP = 0.25, 0.3, 0.3


def configure():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def occupancy(path, warm=180.0, horizon=900.0):
    S = [s for s in json.load(open(path))["samples"] if warm <= s["t"] <= horizon and "usage" in s]
    return 100 * sum(s["usage"] for s in S) / len(S)


def collect():
    reuse, occ = {}, {}
    for name in ORDER:
        reuse[name], occ[name] = {}, {}
        for lab, f in LOADS:
            p = "mc-f%s-%s.json" % (f, TAG[name])
            reuse[name][lab] = summarize(json.load(open(p)), 180.0)["hit_pct"]
            occ[name][lab] = occupancy(p)
    return reuse, occ


def panel(ax, data, ylabel, ylim, yticks, label_size, tick_size):
    x, heights, colors = [], [], []
    centers = []
    curr = 0.0
    for lab, _ in LOADS:
        first = curr
        for i, name in enumerate(ORDER):
            x.append(curr)
            heights.append(data[name][lab])
            colors.append(PALETTE[i % len(PALETTE)])
            curr += PITCH
        centers.append((first + curr - PITCH) / 2)
        curr += GAP
    ax.bar(x, heights, alpha=0.95, color=colors, edgecolor="black",
           width=BAR_W, linewidth=0.8, zorder=3)
    ax.set_ylabel(ylabel, fontsize=label_size, fontweight="bold", color="black")
    ax.set_xticks(centers)
    ax.set_xticklabels([l for l, _ in LOADS], fontsize=label_size, fontweight="bold", color="black")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlim(min(x) - 0.35, max(x) + 0.35)
    ax.tick_params(axis="y", labelsize=tick_size)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")
        t.set_color("black")
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)


# The legend is written as an explicit grid so the reading order is the one intended.
# Matplotlib fills a legend column by column, so the grid is flattened that way and
# short rows are padded with invisible entries.
LEGEND_GRID = [[0, 1, 2, 3, 4]]


def make_handles():
    ncol = len(LEGEND_GRID[0])
    handles = []
    for col in range(ncol):
        for row in LEGEND_GRID:
            i = row[col]
            handles.append(Patch(facecolor="none", edgecolor="none", label=" ") if i is None
                           else Patch(facecolor=PALETTE[i % len(PALETTE)], edgecolor="black",
                                      label=ORDER[i]))
    return handles, ncol


def legend(fig, ax, size, legend_y=0.905):
    handles, ncol = make_handles()
    lg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, legend_y),
                    ncol=ncol, fontsize=size, frameon=True,
                    columnspacing=0.95, handletextpad=0.5)
    for t in lg.get_texts():
        t.set_fontweight("bold")


def legend_in(ax, size):
    """Legend inside the panel, upper right, where the bars leave the plot empty."""
    handles, ncol = make_handles()
    lg = ax.legend(handles=handles, loc="upper right", ncol=ncol, fontsize=size,
                   frameon=True, columnspacing=0.9, handletextpad=0.45,
                   labelspacing=0.32, borderpad=0.4, handlelength=1.5)
    lg.get_frame().set_edgecolor("#bbbbbb")
    for t in lg.get_texts():
        t.set_fontweight("bold")


def save(fig, stem):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (stem, ext)), format=ext, dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print("wrote %s.{pdf,png,svg}" % os.path.join(OUT, stem))
    plt.close(fig)


def main():
    configure()
    reuse, occ = collect()

    fig, axs = plt.subplots(1, 2, figsize=(16.0, 3.5))
    panel(axs[0], reuse, "Prefix Reuse (%)", (0, 40), [0, 20, 40], 24, 20)
    panel(axs[1], occ, "KV Pool Occ (%)", (0, 100), [0, 50, 100], 24, 20)
    legend(fig, axs[0], 15, 0.855)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.21, top=0.80, wspace=0.16)
    save(fig, "fig-cache")

    fig, ax = plt.subplots(figsize=(11.4, 7.1))
    panel(ax, reuse, "Follow-up prefix reuse (%)", (0, 40), [0, 10, 20, 30, 40], 30, 24)
    legend(fig, ax, 11, 0.805)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.16, top=0.78)
    save(fig, "fig-reuse")

    for lab, _ in LOADS:
        print(lab, "  ".join("%s %.1f/%.0f%%" % (n, reuse[n][lab], occ[n][lab]) for n in ORDER))


if __name__ == "__main__":
    main()
