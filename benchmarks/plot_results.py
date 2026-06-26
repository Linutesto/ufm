#!/usr/bin/env python3
"""
Generate figures from results/summary.json.

Outputs (results/):
  fig_vram_vs_experts.png   — peak VRAM as the expert bank grows; the OOM wall.
  fig_throughput.png        — tokens/sec per mode (log scale).
  fig_vram_over_time.png    — VRAM trace for one UFM run (bounded sawtooth), if telemetry exists.

Style is dark with green/pink accents to match yandesbiens.com. No claims are
drawn on the charts beyond what the data shows; OOM is marked explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

BG = "#0b0f10"; FG = "#d7dadc"; GRID = "#23292b"
GREEN = "#22c55e"; PINK = "#ff5fa2"; AMBER = "#f5a524"; BLUE = "#5fb2ff"
MODE_COLOR = {"baseline": PINK, "naive_offload": AMBER, "ufm": GREEN}
MODE_LABEL = {"baseline": "baseline (all experts on GPU)",
              "naive_offload": "naive CPU offload (.cuda()/.cpu() per call)",
              "ufm": "UFM (paged, LRU keep-hot)"}


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, pad=12)
    ax.set_xlabel(xlabel, color=FG); ax.set_ylabel(ylabel, color=FG)
    ax.tick_params(colors=FG); ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)


def _load():
    with open(RESULTS / "summary.json") as f:
        return json.load(f)


def by_mode(results, key):
    out = {}
    for r in results:
        out.setdefault(r["mode"], []).append(r)
    for m in out:
        out[m].sort(key=lambda r: r["num_experts"])
    return out


def fig_vram(data):
    res = data["results"]; vram_total = data["env"]["vram_total_gb"]
    grouped = by_mode(res, "peak_vram_gb")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for mode, rows in grouped.items():
        xs = [r["num_experts"] for r in rows if r["status"] == "ok"]
        ys = [r["peak_vram_gb"] for r in rows if r["status"] == "ok"]
        if xs:
            ax.plot(xs, ys, "o-", color=MODE_COLOR[mode], label=MODE_LABEL[mode], lw=2, ms=6)
        # mark OOM points
        for r in rows:
            if r["status"] == "oom":
                ax.scatter([r["num_experts"]], [vram_total], color=MODE_COLOR[mode],
                           marker="X", s=160, zorder=5)
                ax.annotate("OOM", (r["num_experts"], vram_total), color=MODE_COLOR[mode],
                            fontsize=11, ha="center", va="bottom", xytext=(0, 8),
                            textcoords="offset points", fontweight="bold")
    ax.axhline(vram_total, color=BLUE, ls="--", lw=1.2, label=f"GPU VRAM limit ({vram_total} GB)")
    # secondary x: expert bank size
    banks = {r["num_experts"]: r["expert_bank_gb"] for r in res}
    _style(ax, "Peak VRAM vs. expert-bank size", "number of experts", "peak VRAM (GB)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="upper left")
    xs_all = sorted(banks)
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(xs_all)
    ax2.set_xticklabels([f"{banks[x]:.0f}" for x in xs_all], color=FG, fontsize=8)
    ax2.set_xlabel("expert bank (GB, fp32)", color=FG, fontsize=9)
    ax2.tick_params(colors=FG)
    fig.tight_layout()
    out = RESULTS / "fig_vram_vs_experts.png"
    fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    return out


def fig_throughput(data):
    res = data["results"]
    grouped = by_mode(res, "tokens_per_s")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
    for mode, rows in grouped.items():
        xs = [r["num_experts"] for r in rows if r["tokens_per_s"]]
        ys = [r["tokens_per_s"] for r in rows if r["tokens_per_s"]]
        if xs:
            ax.plot(xs, ys, "o-", color=MODE_COLOR[mode], label=MODE_LABEL[mode], lw=2, ms=6)
    ax.set_yscale("log")
    _style(ax, "Throughput (higher is better)", "number of experts", "tokens / sec (log)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    out = RESULTS / "fig_throughput.png"
    fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    return out


def fig_vram_time(data):
    # pick the largest UFM run that succeeded
    ufm = [r for r in data["results"] if r["mode"] == "ufm" and r["status"] == "ok"]
    if not ufm:
        return None
    n = max(r["num_experts"] for r in ufm)
    csv_path = RESULTS / f"telemetry_ufm_{n:04d}.csv"
    if not csv_path.exists():
        return None
    import csv as _csv
    t, g = [], []
    with open(csv_path) as f:
        for row in _csv.DictReader(f):
            t.append(float(row["t"])); g.append(float(row["gpu_alloc_gb"]))
    if not t:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=BG)
    ax.plot(t, g, color=GREEN, lw=1.6)
    ax.axhline(data["env"]["config"]["ufm_target_gb"], color=PINK, ls="--", lw=1.2,
               label=f"UFM target ({data['env']['config']['ufm_target_gb']} GB)")
    ax.axhline(data["env"]["vram_total_gb"], color=BLUE, ls="--", lw=1.0,
               label=f"GPU limit ({data['env']['vram_total_gb']} GB)")
    _style(ax, f"UFM VRAM over time — {n} experts ({data['results'][0].get('expert_bank_gb','?')} GB bank)",
           "time (s)", "VRAM allocated (GB)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.tight_layout()
    out = RESULTS / "fig_vram_over_time.png"
    fig.savefig(out, dpi=150, facecolor=BG); plt.close(fig)
    return out


def main():
    data = _load()
    outs = [fig_vram(data), fig_throughput(data), fig_vram_time(data)]
    for o in outs:
        if o:
            print(f"wrote {o}")


if __name__ == "__main__":
    main()
