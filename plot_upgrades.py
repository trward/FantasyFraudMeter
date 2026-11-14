#!/usr/bin/env python3
"""Figures for the methodology paper: Monte Carlo significance + Pythagorean cross-check."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description="Significance + cross-check figures")
ap.add_argument("--data", default="fraud_data.json")
ap.add_argument("--upgrades", default="upgrades.json")
args = ap.parse_args()

D = json.load(open(args.data))
U = json.load(open(args.upgrades))
names = D["names"]
Fm = np.array(D["F_margin"])
Fc = np.array(D["F_count"])
z = np.array(U["z"])
p = np.array(U["p"])
null_mu = np.array(U["null_mean"])
null_sd = np.array(U["null_std"])
pyth_gap = np.array(U["pyth_gap"])
pyth_wins = np.array(U["pyth_wins"])
W_obs = np.array(U["W_obs"])
N = len(names)

short = {
    "Olivia's Team": "Olivia", "Olivia's Dad": "David",
    "Becca's Boyfriend": "Chris", "Champ? Is Champ There?": "Wade",
    "Becca\u2019s Baddies": "Becca", "Mel's Magic Men": "Mel",
    "Lincoln's Little Leaguers": "Cory", "Hot boys, tight pants": "Hailey",
    "Corrupt Commissioner": "Joe", "The Cleaning Ladies": "Taylor",
    "Maye the force be with You": "Gideon", "Seymour TDs": "Seymour",
}
labels = [short[nm] for nm in names]
order = np.argsort(Fm)

# ---- Fig: forest plot of fraud coefficient with Monte Carlo null ----
fig, ax = plt.subplots(figsize=(8.2, 6.0), facecolor="white")
ypos = np.arange(N)[::-1]
for rank, i in enumerate(order):
    y = ypos[rank]
    sig = p[i] < 0.05
    col = "#c0392b" if Fm[i] < 0 else "#2f6cb3"
    # null 95% interval
    lo, hi = null_mu[i] - 1.96 * null_sd[i], null_mu[i] + 1.96 * null_sd[i]
    ax.plot([lo, hi], [y, y], color="#999", lw=1.4, zorder=2)
    ax.plot([null_mu[i]], [y], "_", color="#999", ms=9, zorder=3)
    ax.plot([Fm[i]], [y], "o", ms=10, color=col,
            markeredgecolor="black" if sig else "none", markeredgewidth=1.2, zorder=4)
    star = "*" if p[i] < 0.05 else ("." if p[i] < 0.10 else "")
    ax.text(Fm[i] + 0.12, y, f"{Fm[i]:+.2f}{star}", va="center", fontsize=9,
            color="black")
ax.axvline(0, color="#333", lw=1.0, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels([labels[i] for i in order], fontsize=10)
ax.set_xlabel("Margin-weighted fraud coefficient  (dot = observed, _ = null mean, line = null 95% CI)")
ax.set_title("Fraud coefficient vs. Monte Carlo schedule-shuffle null\n"
             "(B=10,000; * p<0.05, . p<0.10)", fontsize=11)
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("fig_significance.png", dpi=150, facecolor="white")
print("saved fig_significance.png")

# ---- Fig: Pythagorean cross-check scatter ----
fig2, ax2 = plt.subplots(figsize=(6.6, 5.8), facecolor="white")
ax2.scatter(pyth_gap, Fm, s=90, c="#6a4c93", edgecolor="black", zorder=3)
for i in range(N):
    ax2.annotate(labels[i], (pyth_gap[i], Fm[i]), textcoords="offset points",
                 xytext=(8, 5), fontsize=9, color="#333")
lim = max(np.abs(pyth_gap).max(), np.abs(Fm).max()) * 1.15
ax2.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
ax2.axvline(0, color="#999", lw=0.6)
ax2.axhline(0, color="#999", lw=0.6)
ax2.set_xlabel("Pythagorean gap  (expected wins from PF/PA $- $ actual wins)")
ax2.set_ylabel("Margin-weighted fraud coefficient")
ax2.set_title("Pythagorean expectation vs. fraud coefficient\n"
              f"(Pearson r = {U['corr_pyth_F']:.3f})", fontsize=11)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("fig_pythagorean.png", dpi=150, facecolor="white")
print("saved fig_pythagorean.png")
