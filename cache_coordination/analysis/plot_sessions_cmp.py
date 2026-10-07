"""Per-session timelines, one panel per scheduling configuration.

Each row is one session and the clock starts at that session's own first arrival, so
a row reads as what one user experienced rather than where the replay happened to be.
Rows keep the same identity across panels and are ordered by how long the session took
under FCFS, which turns the baseline panel into a smooth wedge that the other panels
either flatten or do not.

Every turn is drawn as the time it waits in the engine's queue and then the time it is
served; prefill and decode are merged because prefill is under two seconds and would be
invisible at this scale. Between two turns the user's own gap is drawn in the palest
colour, so the three colours form a ramp by how much the engine is doing for the session.

usage (inside timeline/): plot_sessions_cmp.py scbench | mooncake
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_GAP = "#D8E4EC"     # palest: the session is out of the system
C_WAIT = "#F6B9AF"    # a light tint of the salmon used for queue wait elsewhere
C_SERVE = "#2E5A6B"   # the only dark colour, so service reads first
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")

FIGS = {
    "scbench": {
        "stem": "fig-sessions-scbench",
        "panels": [("FCFS", "n60win60/baseline.json"),
                   ("Cache-Aware", "n60win60/kvaware.json"),
                   ("Cache-Aware + Guard", "n60win60/kvp-W40.json")],
        "multi_only": False,
        "height": 7.8,
    },
    "mooncake": {
        "stem": "fig-sessions-mooncake",
        "panels": [("FCFS", "h100mc/mc-f0.75-baseline.json"),
                   ("Cache-Aware", "h100mc/mc-f0.75-kvaware-nh.json"),
                   ("Cache-Aware + Guard", "h100mc/mc-f0.75-W40.json"),
                   ("Session-Aware", "h100mc/mc-f0.75-retainU.json"),
                   ("Session-Aware + Cache-Aware + Guard", "h100mc/mc-f0.75-W40-retain.json")],
        "multi_only": True,
        "compress": True,
        "height": 12.0,

    },
}


def configure():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def sessions(path):
    by = defaultdict(list)
    for d in json.load(open(path))["records"]:
        by[d["inst"]].append(d)
    for v in by.values():
        v.sort(key=lambda d: d["r"])
    return by


def compressed(turns):
    """Turns laid end to end with the user's gaps removed.

    Each row then measures only the time the session spent inside the engine, which is
    the quantity a scheduler can actually change; on the replay clock that time is a few
    percent of a row and no difference between configurations is visible.
    """
    wait, serve = [], []
    x = 0.0
    for d in turns:
        a = d["t_arrive"]
        ds = max(d["decode_start"], a)
        de = max(d["decode_end"], ds)
        if ds > a:
            wait.append((x, ds - a))
            x += ds - a
        if de > ds:
            serve.append((x, de - ds))
            x += de - ds
    return wait, serve, [], x


def spans_of(turns, relative=False):
    """(wait, serve, gap) segments on the replay clock, or relative to the session start."""
    t0 = turns[0]["t_arrive"] if relative else 0.0
    wait, serve, gap = [], [], []
    for j, d in enumerate(turns):
        a = d["t_arrive"] - t0
        ds = max(d["decode_start"] - t0, a)
        de = max(d["decode_end"] - t0, ds)
        if ds > a:
            wait.append((a, ds - a))
        if de > ds:
            serve.append((ds, de - ds))
        if j + 1 < len(turns):
            nxt = turns[j + 1]["t_arrive"] - t0
            if nxt > de:
                gap.append((de, nxt - de))
    return wait, serve, gap, de


def draw(ax, by, order, title, xmax, label_size, tick_size, compress=False):
    h = 0.82
    rows = {"wait": {}, "serve": {}, "gap": {}}
    for y, inst in enumerate(order):
        if inst not in by:
            continue
        w, s, g, _ = (compressed if compress else spans_of)(by[inst])
        rows["wait"][y], rows["serve"][y], rows["gap"][y] = w, s, g
    for key, color in (("gap", C_GAP), ("wait", C_WAIT), ("serve", C_SERVE)):
        for y, segs in rows[key].items():
            if segs:
                ax.broken_barh(segs, (y - h / 2, h), facecolors=color, edgecolor="none")
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, xmax)
    ax.set_yticks([])
    ax.set_ylabel("Sessions", fontsize=label_size - 3, fontweight="bold",
                  color="black", labelpad=10)
    ax.set_title(title, loc="left", fontsize=label_size, fontweight="bold", color="black", pad=5)
    ax.tick_params(axis="x", labelsize=tick_size)
    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
        t.set_color("black")
    ax.grid(which="major", axis="x", linestyle="-", linewidth=0.5, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "scbench"
    cfg = FIGS[which]
    configure()

    runs = [(name, sessions(path)) for name, path in cfg["panels"]]
    base = runs[0][1]
    keep = [i for i in base if not cfg["multi_only"] or len(base[i]) > 1]
    comp = cfg.get("compress", False)
    if comp:
        # ordered by how long the session sat in the engine under FCFS, so the baseline
        # panel is a smooth wedge and a configuration either flattens it or does not
        order = sorted(keep, key=lambda i: compressed(base[i])[3])
        xmax = q([compressed(base[i])[3] for i in keep], 0.99) * 1.05
    else:
        order = sorted(keep, key=lambda i: base[i][0]["t_arrive"])
        xmax = cfg.get("xmax") or max(spans_of(by[i])[3] for _, by in runs
                                      for i in order if i in by) * 1.01
    print("%s: %d rows, x to %.0f s" % (which, len(order), xmax))
    for name, by in runs:
        sel = [(compressed(by[i])[3] if comp else spans_of(by[i], True)[3])
               for i in order if i in by]
        print("  %-36s %s p50 %6.1f  p95 %6.1f  mean %6.1f"
              % (name, "time in system" if comp else "session span",
                 q(sel, .5), q(sel, .95), sum(sel) / len(sel)))

    n = len(runs)
    fig, axs = plt.subplots(n, 1, figsize=(15.0, cfg["height"]),
                            gridspec_kw={"hspace": 0.42})
    lab = 18 if n > 3 else 21
    for ax, (name, by) in zip(axs, runs):
        draw(ax, by, order, name, xmax, lab, 17, comp)
    axs[-1].set_xlabel("Time In System (s)" if comp else "Time (s)", fontsize=lab + 2, fontweight="bold", color="black")

    handles = [Patch(facecolor=C_WAIT, edgecolor="black", label="Queue wait"),
               Patch(facecolor=C_SERVE, edgecolor="black", label="Service (prefill + decode)")]
    if not comp:                       # the compressed view has no gaps left to label
        handles.append(Patch(facecolor=C_GAP, edgecolor="black",
                             label="User gap (thinking or away)"))
    lg = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.925),
                    ncol=len(handles), fontsize=17, frameon=True, columnspacing=0.95,
                    handletextpad=0.5)
    for t in lg.get_texts():
        t.set_fontweight("bold")
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.105, top=0.895)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (cfg["stem"], ext)), format=ext, dpi=300)
    print("wrote", os.path.join(OUT, cfg["stem"]))


if __name__ == "__main__":
    main()
