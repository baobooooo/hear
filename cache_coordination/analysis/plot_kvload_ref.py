"""KV load over time at one load level, one line per configuration.

Left:   how full the GPU KV pool is. The configurations sit on top of each other, which
        is the control for the obvious objection: the reuse gain is not bought with
        memory.
Middle: KV pulled back from the CPU tier, cumulative. This is the retention mechanism
        doing its work and it separates the configurations by about a factor of two.
Right:  requests waiting in the engine. The gate that conditions retention reads exactly
        this number.

Style follows REFLECT_2AFC_barplot.py; the fill palette of the bar figures is darkened
so the same configuration keeps its hue but reads as a line. Run inside timeline/h100mc.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")
KV_TOK = 147456
LOAD = sys.argv[1] if len(sys.argv) > 1 else "0.75"
WARM, XMAX = 180.0, 1500.0

ORDER = [("FCFS", "baseline", "#4E8C82", "-"),
         ("Cache-Aware", "kvaware-nh", "#4A86C4", "-"),
         ("Cache-Aware + Guard", "W40", "#D4573F", (0, (5, 2))),
         ("Session-Aware", "retainU", "#E08A32", "-"),
         ("Session-Aware + Cache-Aware + Guard", "W40-retain", "#A8813F", (0, (5, 2)))]


def configure():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"




def load(tag):
    S = json.load(open("mc-f%s-%s.json" % (LOAD, tag)))["samples"]
    S = [s for s in S if "usage" in s and s["t"] <= XMAX]
    t = [s["t"] for s in S]
    base = S[0]["l2_in"]
    restored = [(s["l2_in"] - base) / KV_TOK / 1e6 for s in S]
    # occupancy and queue length are sampled every 0.47 s and swing between batches;
    # a 30 s rolling mean is what a reader can actually compare across five lines
    k = int(30 / max(0.1, t[1] - t[0])) if len(t) > 1 else 1
    smooth = lambda v: [sum(v[max(0, i - k + 1):i + 1]) / len(v[max(0, i - k + 1):i + 1])
                        for i in range(len(v))]
    return t, smooth([100 * s["usage"] for s in S]), restored, smooth([s["waiting"] for s in S])


def panel(ax, xs, ys, colors, styles, labels, ylabel, ylim, yticks, label_size, tick_size):
    ax.axvspan(0, WARM, color="#f0f0ee", zorder=0)
    for x, y, c, ls, lab in zip(xs, ys, colors, styles, labels):
        ax.plot(x, y, color=c, linewidth=2.0, linestyle=ls, label=lab, zorder=3)
    ax.set_ylabel(ylabel, fontsize=label_size, fontweight="bold", color="black")
    ax.set_xlabel("Time (s)", fontsize=label_size, fontweight="bold", color="black")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.tick_params(labelsize=tick_size)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight("bold")
        lb.set_color("black")
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def main():
    configure()
    D = [load(tag) for _, tag, _, _ in ORDER]
    labels = [n for n, _, _, _ in ORDER]
    colors = [c for _, _, c, _ in ORDER]
    styles = [ls for _, _, _, ls in ORDER]
    ts = [d[0] for d in D]
    for (n, _, _, _), d in zip(ORDER, D):
        print("%-36s occ mean %4.1f%%   restored %5.2fM tok   waiting mean %4.1f"
              % (n, sum(d[1]) / len(d[1]), d[2][-1], sum(d[3]) / len(d[3])))

    fig, axs = plt.subplots(1, 3, figsize=(17.0, 3.9))
    panel(axs[0], ts, [d[1] for d in D], colors, styles, labels,
          "GPU KV Pool (%)", (0, 100), [0, 50, 100], 21, 17)
    panel(axs[1], ts, [d[2] for d in D], colors, styles, labels,
          "KV Restored (M tok)", (0, 1.25), [0, 0.5, 1.0], 21, 17)
    panel(axs[2], ts, [d[3] for d in D], colors, styles, labels,
          "Requests Waiting", (0, 120), [0, 60, 120], 21, 17)

    lg = fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center",
                    bbox_to_anchor=(0.5, 0.905), ncol=5, fontsize=16, frameon=True,
                    columnspacing=0.95, handletextpad=0.5, handlelength=1.8)
    for t in lg.get_texts():
        t.set_fontweight("bold")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.19, top=0.80, wspace=0.26)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, "fig-kvload.%s" % ext), format=ext, dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    print("wrote", os.path.join(OUT, "fig-kvload.{pdf,png,svg}"))


if __name__ == "__main__":
    main()
