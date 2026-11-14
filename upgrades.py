#!/usr/bin/env python3
"""Literature-grounded upgrades to the Fraud Coefficient.

Adds, on top of compute_fraud.py output:
  1. Monte Carlo schedule-shuffle null distribution  (Getty et al. 2018, Sec 2.3)
     -> z-score + two-sided p-value for each team's fraud coefficient,
        isolating the schedule-luck component (the 'ideal' term is
        schedule-independent; only the actual pairing is randomized).
  2. Pythagorean expectation cross-check             (Bill James; Carroll et al. 1988)
     -> expected wins from aggregate points for/against, gap vs actual.
  3. Noll-Scully competitive balance                 (Lopez et al. 2018, Sec 1.1.1)
     -> observed vs idealized (chance) std of team win totals.

Usage:
    python upgrades.py --data fraud_data.json --league league_data.json \
                       [--b 10000] [--seed 0] [--out upgrades.json]
"""
import argparse
import json
import numpy as np


def random_perfect_matching(n, rng):
    """Return a list of n/2 (a, b) index pairs, a random perfect matching."""
    idx = rng.permutation(n)
    return list(zip(idx[0::2], idx[1::2]))


def main():
    ap = argparse.ArgumentParser(description="Literature upgrades for the Fraud Coefficient")
    ap.add_argument("--data", default="fraud_data.json")
    ap.add_argument("--league", default="league_data.json")
    ap.add_argument("--b", type=int, default=10000, help="Monte Carlo replications")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pyth-exponent", type=float, default=2.0)
    ap.add_argument("--out", default="upgrades.json")
    args = ap.parse_args()

    D = json.load(open(args.data))
    L = json.load(open(args.league))
    names = D["names"]
    N, W = D["N"], D["N_weeks"]
    S = np.array(D["S"])                 # (N, W), NaN = bye
    e = np.array(D["e_weekly"])          # (N, W) expected win rate vs random opp
    Wres = np.array(D["Wres"])           # (N, W) +1/-1/NaN
    F_margin = np.array(D["F_margin"])
    F_count = np.array(D["F_count"])
    wins = (Wres + 1) / 2                # 1/0/NaN

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Monte Carlo schedule-shuffle null
    #    E_i = sum_w e_i(w) is schedule-independent (fixed).
    #    F_i = E_i - W_i where W_i = sum_w win_i(w) is the actual win total.
    #    Under random scheduling, W_i has a null distribution; we resample
    #    a random perfect matching each week (scores held fixed).
    # ------------------------------------------------------------------
    E = np.nansum(e, axis=1)             # (N,) fixed ideal total
    W_obs = np.nansum(wins, axis=1)      # (N,) actual win total
    F_obs = E - W_obs                    # == F_margin (sanity)
    assert np.allclose(F_obs, F_margin, atol=1e-9), "F_obs != F_margin"

    W_sim = np.zeros((args.b, N))
    active = ~np.isnan(S)
    for b in range(args.b):
        for w in range(W):
            act = np.where(active[:, w])[0]
            if len(act) < 2 or len(act) % 2:
                continue
            for a, bb in random_perfect_matching(len(act), rng):
                ia, ib = act[a], act[bb]
                if S[ia, w] > S[ib, w]:
                    W_sim[b, ia] += 1
                elif S[ib, w] > S[ia, w]:
                    W_sim[b, ib] += 1
                # tie: no win credited to either (0.5 each would also be fine)
    F_sim = E[None, :] - W_sim           # (B, N)
    null_mean = F_sim.mean(axis=0)
    null_std = F_sim.std(axis=0, ddof=1)
    z = (F_obs - null_mean) / null_std
    # two-sided p-value
    p = (2 * np.minimum((F_sim < F_obs[None, :]).mean(axis=0),
                        (F_sim > F_obs[None, :]).mean(axis=0))).clip(0, 1)

    print("=== Monte Carlo schedule-shuffle null (B=%d) ===" % args.b)
    print(f"{'Team':<28} {'F_obs':>7} {'null_mu':>8} {'null_sd':>8} {'z':>7} {'p':>7}")
    for i, nm in enumerate(names):
        print(f"{nm:<28} {F_obs[i]:>7.2f} {null_mean[i]:>8.2f} {null_std[i]:>8.2f} "
              f"{z[i]:>7.2f} {p[i]:>7.3f}")

    # ------------------------------------------------------------------
    # 2. Pythagorean expectation cross-check (Bill James)
    # ------------------------------------------------------------------
    g = args.pyth_exponent
    PF = np.zeros(N)
    for i in range(N):
        for w in range(W):
            if not np.isnan(S[i, w]):
                PF[i] += S[i, w]
    # PA = sum of the scores of the opponents actually played, from league data
    id2name = {}
    for wk in [str(w) for w in range(1, W + 1)]:
        for tid, t in L["weeks"][wk]["teams"].items():
            id2name[str(tid)] = t["name"].replace("Team(", "").rstrip(" )")
    name2idx = {nm: i for i, nm in enumerate(names)}
    PA = np.zeros(N)
    for wi, wk in enumerate([str(w) for w in range(1, W + 1)]):
        for m in L["weeks"][wk]["matchups"]:
            if m["home"] is None or m["away"] is None:
                continue
            hn, an = id2name.get(str(m["home"])), id2name.get(str(m["away"]))
            if hn is None or an is None or m["home_score"] is None or m["away_score"] is None:
                continue
            hi, ai = name2idx[hn], name2idx[an]
            PA[hi] += float(m["away_score"])
            PA[ai] += float(m["home_score"])
    pyth_pct = PF ** g / (PF ** g + PA ** g)
    pyth_wins = pyth_pct * W
    pyth_gap = pyth_wins - W_obs        # + = better record than PF/PA implies
    print("\n=== Pythagorean expectation (g=%.1f) ===" % g)
    print(f"{'Team':<28} {'PF':>8} {'PA':>8} {'PythW':>7} {'ActW':>5} {'gap':>6}")
    for i, nm in enumerate(names):
        print(f"{nm:<28} {PF[i]:>8.1f} {PA[i]:>8.1f} {pyth_wins[i]:>7.2f} "
              f"{W_obs[i]:>5.1f} {pyth_gap[i]:>6.2f}")
    print("corr(PythGap, F_margin) =",
          round(float(np.corrcoef(pyth_gap, F_margin)[0, 1]), 3))

    # ------------------------------------------------------------------
    # 3. Noll-Scully competitive balance (Lopez et al. 2018)
    # ------------------------------------------------------------------
    win_totals = W_obs
    obs_sd = win_totals.std(ddof=1)
    ideal_sd = np.sqrt(W) / 2.0         # chance: each game Bernoulli(0.5)
    noll_scully = obs_sd / ideal_sd
    print("\n=== Noll-Scully competitive balance ===")
    print(f"observed sd of win totals = {obs_sd:.3f}")
    print(f"idealized (chance) sd     = {ideal_sd:.3f}")
    print(f"Noll-Scully               = {noll_scully:.3f}  "
          f"(>1 = more imbalanced than chance)")

    json.dump({
        "B": args.b, "seed": args.seed, "pyth_exponent": g,
        "E_ideal_total": E.tolist(),
        "W_obs": W_obs.tolist(),
        "null_mean": null_mean.tolist(), "null_std": null_std.tolist(),
        "z": z.tolist(), "p": p.tolist(),
        "PF": PF.tolist(), "PA": PA.tolist(),
        "pyth_wins": pyth_wins.tolist(), "pyth_gap": pyth_gap.tolist(),
        "noll_scully": noll_scully, "obs_sd": obs_sd, "ideal_sd": ideal_sd,
        "corr_pyth_F": float(np.corrcoef(pyth_gap, F_margin)[0, 1]),
    }, open(args.out, "w"), indent=1)
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
