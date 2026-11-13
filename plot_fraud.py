#!/usr/bin/env python3
"""Plot the Fraud Meter (margin-weighted) from compute_fraud.py output.

Usage:
    python plot_fraud.py --data fraud_data.json [--out fraud_meter_3.png]
                         [--report-dir DIR] [--zone-bounds -1,0,1]

    --out          main dark two-panel composite (meter + weekly trajectory)
    --report-dir   if given, also writes light-themed figures for a report:
                   fig_meter_report.png, fig_timeseries_report.png,
                   fig_compare.png, fig_heatmap.png
    --zone-bounds  three comma-separated x-axis bounds for the four zones
                   (default -1,0,1; in cumulative coefficient units)
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ZONE_LABELS = ["!!FRAUD ZONE!!", "Sus", "Chillin", "Unlucky asf"]
ZONE_COLORS = ["#c0392b", "#b7950b", "#2e8b57", "#2f6cb3"]
ZONE_FILLS = ["#d64545", "#e8c84a", "#5cb85c", "#4a90d9"]


def short_names(names, owners):
    """Owner first names; full name on collision; team name as last resort."""
    first = {}
    for nm in names:
        o = owners.get(nm) or {}
        fn = (o.get("firstName") or "").strip()
        ln = (o.get("lastName") or "").strip()
        first[nm] = fn or nm
    used = {}
    out = []
    for nm in names:
        lab = first[nm]
        if lab in used:
            o = owners.get(nm) or {}
            lab = f"{first[nm]} {o.get('lastName', '')}".strip() or nm
        used[lab] = True
        out.append(lab)
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot the Fraud Meter")
    ap.add_argument("--data", default="fraud_data.json")
    ap.add_argument("--out", default="fraud_meter_3.png")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--zone-bounds", default="-1,0,1")
    args = ap.parse_args()

    D = json.load(open(args.data))
    names = D["names"]
    Fm = np.array(D["F_margin"])
    Fc = np.array(D["F_count"])
    delta = np.array(D["delta_weekly"]).T          # (N, W)
    cum = np.nancumsum(delta, axis=1)
    N, NW = len(names), D["N_weeks"]
    labels = short_names(names, D.get("owners", {}))

    zb = [float(x) for x in args.zone_bounds.split(",")]
    assert len(zb) == 3, "--zone-bounds needs exactly 3 values"
    zmin, zmax = min(Fm.min(), zb[0] - 1.2), max(Fm.max(), zb[2] + 1.2)
    ZONES = [
        (zmin - 0.3, zb[0], ZONE_FILLS[0], ZONE_LABELS[0]),
        (zb[0], zb[1], ZONE_FILLS[1], ZONE_LABELS[1]),
        (zb[1], zb[2], ZONE_FILLS[2], ZONE_LABELS[2]),
        (zb[2], zmax + 0.3, ZONE_FILLS[3], ZONE_LABELS[3]),
    ]
    zone_of = lambda x: 0 if x < zb[0] else 1 if x < zb[1] else 2 if x < zb[2] else 3
    order = np.argsort(Fm)  # most fraudulent first

    def draw_meter(ax, dark):
        ax.set_facecolor("#1a2027" if dark else "white")
        top, bot = N - 0.5, -0.5
        for x0, x1, c, lab in ZONES:
            ax.add_patch(Rectangle((x0, bot), x1 - x0, top - bot, facecolor=c,
                                   alpha=0.16, edgecolor="none", zorder=0))
        ax.axvline(0, color="#8899aa", lw=1.2, ls="--", alpha=0.75, zorder=1)
        for rank, i in enumerate(order):
            x = Fm[i]
            y = N - 1 - rank
            c = ZONE_COLORS[zone_of(x)]
            ax.hlines(y, 0, x, colors=c, lw=2.2, alpha=0.85, zorder=2)
            ax.plot([x], [y], "o", ms=9, color=c,
                    markeredgecolor="white" if dark else "black",
                    markeredgewidth=1.0, zorder=3)
            dx = 0.09 if x >= 0 else -0.09
            ax.text(x + dx, y, f"{x:+.2f}", va="center",
                    ha="left" if x >= 0 else "right", fontsize=9,
                    color="#cfd8e3" if dark else "#333", zorder=4)
        ax.set_yticks(range(N - 1, -1, -1))
        ax.set_yticklabels([labels[i] for i in order],
                           color="white" if dark else "#222", fontsize=10)
        ax.set_ylim(-0.7, N + 0.45)
        ax.set_xlim(zmin - 0.55, zmax + 0.55)
        ticks = np.arange(np.ceil(zmin), np.floor(zmax) + 1)
        ax.set_xticks(ticks)
        ax.tick_params(colors="#9fb4c8" if dark else "#333", length=0)
        for s in ax.spines.values():
            s.set_color("#3a4654" if dark else "#999")
        for x0, x1, c, lab in ZONES:
            ax.text((x0 + x1) / 2, N - 0.18, lab, ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold", color=c)
        ax.set_xlabel("FRAUDULENCE COEFFICIENT  (negative = fraudulent, positive = unlucky)\n"
                      "(science's most accurate representation of integrity and honesty amen)",
                      color="#cfd8e3" if dark else "#222", fontsize=10)
        ax.set_title("FRAUD METER 3.0 \u2014 margin-weighted\n"
                     f"End of Week {NW}  \u00b7  {N} teams  \u00b7  {NW} weeks",
                     color="white" if dark else "#111", fontsize=13,
                     fontweight="bold", pad=26)

    def draw_timeseries(ax, dark):
        ax.set_facecolor("#1a2027" if dark else "white")
        for i in order:
            z = zone_of(Fm[i])
            ax.plot(range(1, NW + 1), cum[i], color=ZONE_COLORS[z], lw=2.0,
                    alpha=0.9, marker="o", markersize=3.5, zorder=3)
        ends = sorted(range(N), key=lambda i: cum[i, -1])
        placed = []
        for i in ends:
            y = cum[i, -1]
            for py in placed:
                if abs(y - py) < 0.28:
                    y = py + (0.28 if y >= py else -0.28)
            placed.append(y)
            z = zone_of(Fm[i])
            ax.annotate(labels[i], (NW + 0.12, y), fontsize=9,
                        color=ZONE_COLORS[z], fontweight="bold", va="center")
        ax.axhline(0, color="#8899aa", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlim(0.8, NW + 1.3)
        ax.set_xticks(range(1, NW + 1))
        ax.tick_params(colors="#9fb4c8" if dark else "#333")
        for s in ax.spines.values():
            s.set_color("#3a4654" if dark else "#999")
        ax.set_xlabel("Week", color="#cfd8e3" if dark else "#222", fontsize=10)
        ax.set_ylabel("Cumulative fraud coefficient",
                      color="#cfd8e3" if dark else "#222", fontsize=10)
        ax.set_title("Cumulative fraud by week\n(red = fraudulent \u2192 blue = unlucky)",
                     color="white" if dark else "#111", fontsize=12, fontweight="bold")
        ax.grid(True, color="#3a4654" if dark else "#ddd", lw=0.5, alpha=0.6)

    # ---- main dark composite ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#101418")
    draw_meter(ax1, dark=True)
    draw_timeseries(ax2, dark=True)
    fig.tight_layout(w_pad=3)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print("saved", args.out)

    if not args.report_dir:
        return

    import os
    os.makedirs(args.report_dir, exist_ok=True)
    r = lambda p: os.path.join(args.report_dir, p)

    figm, axm = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    draw_meter(axm, dark=False)
    figm.tight_layout()
    figm.savefig(r("fig_meter_report.png"), dpi=150, facecolor="white")
    print("saved", r("fig_meter_report.png"))

    figt, axt = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    draw_timeseries(axt, dark=False)
    figt.tight_layout()
    figt.savefig(r("fig_timeseries_report.png"), dpi=150, facecolor="white")
    print("saved", r("fig_timeseries_report.png"))

    fig2, ax = plt.subplots(figsize=(6.4, 5.6), facecolor="white")
    for i in range(N):
        ax.scatter(Fc[i], Fm[i], s=90, color=ZONE_COLORS[zone_of(Fm[i])], zorder=3)
        ax.annotate(labels[i], (Fc[i], Fm[i]), textcoords="offset points",
                    xytext=(8, 5), fontsize=9, color="#333")
    lim = max(np.abs(Fc).max(), np.abs(Fm).max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#999", lw=0.6)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("Fraud coefficient \u2014 count-based (Fraud Meter 2.0)")
    ax.set_ylabel("Fraud coefficient \u2014 margin-weighted (Fraud Meter 3.0)")
    ax.set_title("Count-based vs margin-weighted fraud coefficient\n"
                 f"(Pearson r = {np.corrcoef(Fc, Fm)[0, 1]:.3f})", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(r("fig_compare.png"), dpi=150, facecolor="white")
    print("saved", r("fig_compare.png"))

    fig3, ax3 = plt.subplots(figsize=(7.6, 5.6), facecolor="white")
    d2 = np.nan_to_num(delta)
    im = ax3.imshow(d2, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax3.set_xticks(range(NW), [f"W{w}" for w in range(1, NW + 1)])
    ax3.set_yticks(range(N), labels)
    ax3.set_xlabel("Week")
    for i in range(N):
        for w in range(NW):
            if np.isnan(delta[i, w]):
                ax3.text(w, i, "bye", ha="center", va="center",
                         fontsize=7.5, color="#888")
                continue
            ax3.text(w, i, f"{delta[i, w]:+.2f}", ha="center", va="center",
                     fontsize=7.5,
                     color="black" if abs(delta[i, w]) < 0.7 else "white")
    ax3.set_title("Weekly fraud coefficient \u03b4\u1d62(w) = e\u1d62(w) \u2212 w\u1d62(w)\n"
                  "(red = fraudulent week, blue = unlucky week)", fontsize=11)
    fig3.colorbar(im, ax=ax3, label="weekly fraud")
    fig3.tight_layout()
    fig3.savefig(r("fig_heatmap.png"), dpi=150, facecolor="white")
    print("saved", r("fig_heatmap.png"))


if __name__ == "__main__":
    main()
