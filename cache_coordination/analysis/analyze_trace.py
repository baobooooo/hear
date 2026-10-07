"""Steady-state metrics and time-series curves for trace-driven runs.

Measurement window: turns that ARRIVE in [WARM, horizon] (default WARM=180 s);
the batch is still drained to the end so every windowed turn has a TTFT.
Throughput counts turns that FINISH inside the window.

usage: analyze_trace.py run1.json [run2.json ...] [--plot out.png] [--warm 180]
"""
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
SURFACE, GRID, INK, INK_2 = "#fcfcfb", "#e4e3df", "#1f2328", "#5c5b57"
C_WAIT, C_PREFILL, C_DECODE = "#1AAFC4", "#EE9341", "#2F7F3A"
C_L1, C_L2, C_RC = "#2a78d6", "#1baf7a", "#eb6834"
PALETTE = ["#b5b3ae", "#e0913f", "#a8561c", "#1f2328", "#8C6FD8", "#4f94e0"]
KV_TOK_BYTES = 147456


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else float("nan")


def ttft(d):
    return max(d["decode_start"], d["t_arrive"]) - d["t_arrive"]


def summarize(run, warm):
    R = run["records"]
    meta = run.get("workload") or {}
    H = meta.get("horizon_s", max(d["t_arrive"] for d in R))
    W = [d for d in R if warm <= d["t_arrive"] <= H]
    done = [d for d in R if warm <= d["decode_end"] <= H]
    tt = [ttft(d) for d in W]
    by = defaultdict(list)
    for d in R:
        by[d["inst"]].append(d)
    sess = [v for v in by.values() if warm <= min(x["t_arrive"] for x in v) <= H]
    e2e = [max(x["decode_end"] for x in v) - min(x["t_arrive"] for x in v) for v in sess]
    # per-session service time = sum of its turns' (arrival -> last token), i.e. E2E minus think time
    svc = [sum(x["decode_end"] - x["t_arrive"] for x in v) for v in sess]
    te2e = [d["decode_end"] - d["t_arrive"] for d in W]          # per-turn response time
    kvp = [d["kvp"] for d in W if d.get("kvp")]
    later = [d for d in W if d["r"] > 0]
    out = {
        "label": "%s%s" % (run["group"], " W=%g" % run["protect_s"] if run["group"] == "kvprotect" else ""),
        "mode": meta.get("mode"), "rate": meta.get("rate_per_min"), "n_window": len(W),
        "ttft_mean": sum(tt) / len(tt), "ttft_p50": q(tt, .5), "ttft_p95": q(tt, .95),
        "ttft_p99": q(tt, .99), "ttft_max": max(tt),
        "thr_turns_min": 60 * len(done) / (H - warm),
        "thr_ktok_min": 60 * sum(d["prompt_len"] for d in done) / (H - warm) / 1e3,
        "hit_pct": 100 * sum(d["cached"] for d in later) / max(1, sum(d["prompt_len"] for d in later)),
        "recompute_M": sum(d["prompt_len"] - d["cached"] for d in W) / 1e6,
        "sess_e2e_p50": q(e2e, .5), "sess_e2e_p95": q(e2e, .95), "sess_e2e_mean": sum(e2e) / len(e2e),
        "sess_svc_mean": sum(svc) / len(svc), "sess_svc_p50": q(svc, .5), "sess_svc_p95": q(svc, .95),
        "turn_e2e_mean": sum(te2e) / len(te2e), "turn_e2e_p50": q(te2e, .5), "turn_e2e_p95": q(te2e, .95),
        "n_sess": len(sess),
        "makespan": max(d["decode_end"] for d in R),
        "promoted_pct": 100 * sum(1 for k in kvp if k.get("kvp_promoted")) / len(kvp) if kvp else float("nan"),
        "l2_share_pct": (100 * sum(k.get("kvp_l2_tokens", 0) for k in kvp)
                         / max(1, sum(k.get("kvp_l1_tokens", 0) + k.get("kvp_l2_tokens", 0) for k in kvp))
                         if kvp else float("nan")),
    }
    return out


def curves(runs, names, out, warm):
    fig, axs = plt.subplots(5, 1, figsize=(15, 16), sharex=True, gridspec_kw={"hspace": 0.28})
    fig.patch.set_facecolor(SURFACE)
    for ax in axs:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for i, (run, name) in enumerate(zip(runs, names)):
        col = PALETTE[i % len(PALETTE)]
        S = run.get("samples") or []
        t = [s["t"] for s in S]
        axs[0].plot(t, [s.get("waiting", 0) for s in S], color=col, lw=1.4, label="%s 等待" % name)
        axs[0].plot(t, [s.get("running", 0) for s in S], color=col, lw=1.4, ls="--", label="%s 运行" % name)
        u = [s.get("util", 0) if s.get("util", 0) == s.get("util", 0) else 0.0 for s in S]   # nan -> 0 (sampler timeout)
        k = max(1, int(10 / max(0.1, (t[1] - t[0]) if len(t) > 1 else 0.5)))   # ~10 s rolling mean
        us = [sum(u[max(0, j - k + 1):j + 1]) / len(u[max(0, j - k + 1):j + 1]) for j in range(len(u))]
        axs[1].plot(t, us, color=col, lw=1.4, label=name)
        axs[2].plot(t, [100 * s.get("usage", 0) for s in S], color=col, lw=1.4, label=name)
        # rolling prefix-hit share of prompt tokens (60 s window, by decode start)
        R = sorted(run["records"], key=lambda d: d["decode_start"])
        xs, ys = [], []
        for k in range(0, int(R[-1]["decode_start"]) + 1, 10):
            win = [d for d in R if k - 60 <= d["decode_start"] < k]
            if win:
                xs.append(k)
                ys.append(100 * sum(d["cached"] for d in win) / max(1, sum(d["prompt_len"] for d in win)))
        axs[3].plot(xs, ys, color=col, lw=1.6, label=name)
        l2 = [(s.get("l2_in", 0) - S[0].get("l2_in", 0)) / KV_TOK_BYTES / 1e6 for s in S]
        axs[4].plot(t, l2, color=col, lw=1.6, label=name)
    H = (runs[0].get("workload") or {}).get("horizon_s")
    for ax in axs:
        ax.axvspan(0, warm, color="#eeeeea", zorder=0)
        if H:
            ax.axvline(H, color=INK_2, ls=":", lw=1)
    axs[0].set_ylabel("请求数"); axs[0].legend(ncol=len(runs), frameon=False, fontsize=9, loc="upper right")
    axs[1].set_ylabel("GPU 利用率 %（10 s 均值）"); axs[1].set_ylim(0, 105)
    axs[2].set_ylabel("L1 KV 被运行请求占用 %"); axs[2].set_ylim(0, 105)
    axs[3].set_ylabel("60 s 滚动命中率 %"); axs[3].set_ylim(0, 105)
    axs[4].set_ylabel("累计 L2 回灌 (M tok)")
    axs[4].set_xlabel("时间 (s)；灰底 = 暖机 %g s，点线 = 会话到达截止" % warm)
    for ax in axs[1:]:
        ax.legend(ncol=len(runs), frameon=False, fontsize=9, loc="upper right")
    m = runs[0].get("workload") or {}
    src = "Mooncake（Kimi）对话轨迹 %s%%" % int(100 * m.get("frac", 0)) if m.get("kind") == "mooncake" else "BurstGPT 会话"
    fig.suptitle("轨迹驱动负载：%s，%.0f 会话/分钟，思考中位 %.0f s（真实间隔扣除服务时间）"
                 % (src, m.get("rate_per_min", 0), m.get("think_p50", 0)),
                 x=0.06, ha="left", fontsize=14, color=INK, y=0.995)
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--warm", type=float, default=180.0)
    a = ap.parse_args()
    runs = [json.load(open(p)) for p in a.runs]
    S = [summarize(r, a.warm) for r in runs]
    cols = [("TTFT mean", "ttft_mean", "%.1f"), ("p50", "ttft_p50", "%.1f"), ("p95", "ttft_p95", "%.1f"),
            ("p99", "ttft_p99", "%.1f"), ("max", "ttft_max", "%.1f"), ("turns/min", "thr_turns_min", "%.1f"),
            ("ktok/min", "thr_ktok_min", "%.0f"), ("hit%", "hit_pct", "%.0f"), ("recomp M", "recompute_M", "%.2f"),
            ("sess e2e p50", "sess_e2e_p50", "%.0f"), ("sess p95", "sess_e2e_p95", "%.0f"),
            ("promoted%", "promoted_pct", "%.0f"), ("L2 share%", "l2_share_pct", "%.0f"), ("makespan", "makespan", "%.0f")]
    print("window: arrivals in [%g, horizon] s   (%s, R=%s/min, %d turns in window)"
          % (a.warm, S[0]["mode"], S[0]["rate"], S[0]["n_window"]))
    print("%-18s" % "" + "".join("%13s" % c[0] for c in cols))
    for s in S:
        print("%-18s" % s["label"][:18] + "".join("%13s" % (c[2] % s[c[1]]) for c in cols))
    if a.plot:
        curves(runs, [s["label"] for s in S], a.plot, a.warm)


if __name__ == "__main__":
    main()
