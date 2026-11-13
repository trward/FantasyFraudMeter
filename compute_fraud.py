#!/usr/bin/env python3
"""Compute the Fraud Coefficient for a head-to-head fantasy football league.

For each team and week the meter compares the team's actual head-to-head
result against what a hypothetical round-robin (everyone plays everyone that
week) would have given:

  Count-based (Fraud Meter 2.0):
      ideal_i(w) = #{j != i : s_i(w) > s_j(w)} / (n_active - 1)
      delta_i(w) = ideal_i(w) - win_i(w)

  Margin-weighted (Fraud Meter 3.0):
      p_ij(w) = 1 / (1 + exp(-(s_i(w) - s_j(w)) / K))
      ideal_i(w) = mean_{j != i} p_ij(w)
      delta_i(w) = ideal_i(w) - win_i(w)

Sign convention: positive = unlucky (scored well, lost the matchup),
negative = fraudulent (scored poorly, won the matchup). The season
coefficient is F_i = sum_w delta_i(w).

Both variants are exactly zero-mean across teams in every week in which all
teams played (the sum of p_ij over ordered pairs is n-1, and the sum of
actual wins is n/2), so the meter is centered at 0 by construction.

Teams on a bye in a given week (null score) are excluded from that week.

Usage:
    python compute_fraud.py --data league_data.json [--weeks N] [--k K]
                            [--out fraud_data.json]

    --weeks  completed weeks to use (default: current_week - 1)
    --k      logistic scale parameter in points (default: median absolute
             pairwise margin over the used weeks)
"""
import argparse
import json
import numpy as np


def clean_name(name):
    return name.replace("Team(", "").rstrip(" )")


def main():
    ap = argparse.ArgumentParser(description="Compute the Fraud Coefficient")
    ap.add_argument("--data", default="league_data.json")
    ap.add_argument("--weeks", type=int, default=None,
                    help="completed weeks to use (default: current_week - 1)")
    ap.add_argument("--k", type=float, default=None,
                    help="logistic scale parameter (default: median |margin|)")
    ap.add_argument("--out", default="fraud_data.json")
    args = ap.parse_args()

    d = json.load(open(args.data))
    n = len(d["standings"])
    if args.weeks is None:
        cw = d.get("current_week") or 1
        args.weeks = max(1, cw - 1)
    W = args.weeks
    weeks = [str(w) for w in range(1, W + 1)]

    # ---- score matrix S[i, w]; NaN = bye / no score ----
    names, scores = [], {}
    for wk in weeks:
        if wk not in d["weeks"]:
            raise SystemExit(f"week {wk} missing from {args.data}")
        for tid, t in d["weeks"][wk]["teams"].items():
            nm = clean_name(t["name"])
            if nm not in scores:
                names.append(nm)
                scores[nm] = {}
            sc = t.get("score")
            scores[nm][wk] = float(sc) if sc is not None else float("nan")
    S = np.array([[scores[nm][str(w)] for w in range(1, W + 1)] for nm in names])
    active = ~np.isnan(S)
    n_teams = len(names)

    # ---- scale parameter K: median absolute pairwise margin ----
    if args.k is not None:
        K = args.k
    else:
        margins = []
        for wk in range(W):
            act = np.where(active[:, wk])[0]
            if len(act) < 2:
                continue
            M = S[act, wk][:, None] - S[act, wk][None, :]
            margins.append(M[~np.eye(len(act), dtype=bool)])
        K = float(np.median(np.abs(np.concatenate(margins))))

    # ---- actual results: +1 win, -1 loss, NaN bye ----
    id2name = {}
    for wk in weeks:
        for tid, t in d["weeks"][wk]["teams"].items():
            id2name[str(tid)] = clean_name(t["name"])
    name2idx = {nm: i for i, nm in enumerate(names)}

    Wres = np.full((n_teams, W), np.nan)
    opp = {}
    for wi, wk in enumerate(weeks):
        for m in d["weeks"][wk]["matchups"]:
            if m["home"] is None or m["away"] is None:
                continue
            hn, an = id2name.get(str(m["home"])), id2name.get(str(m["away"]))
            if hn is None or an is None:
                continue
            hs, as_ = m["home_score"], m["away_score"]
            if hs is None or as_ is None:
                continue
            hi, ai = name2idx[hn], name2idx[an]
            if float(hs) > float(as_):
                Wres[hi, wi], Wres[ai, wi] = 1, -1
            elif float(hs) < float(as_):
                Wres[ai, wi], Wres[hi, wi] = 1, -1
            opp[(hi, wi)], opp[(ai, wi)] = an, hn
    wins = (Wres + 1) / 2  # 1 / 0 / 0.5, NaN for byes

    # ---- count-based (Fraud Meter 2.0, normalized to [0,1] per week) ----
    Wstar = np.full((W, n_teams), np.nan)
    for wk in range(W):
        act = np.where(active[:, wk])[0]
        for i in act:
            others = [j for j in act if j != i]
            if not others:
                continue
            Wstar[wk, i] = sum(1 for j in others if S[i, wk] > S[j, wk])
    f_count = Wstar / np.maximum(active.sum(axis=0) - 1, 1)[:, None] - wins.T
    F_count = np.nansum(f_count, axis=0)

    # ---- margin-weighted (Fraud Meter 3.0) ----
    e = np.full((n_teams, W), np.nan)
    P_all = np.full((n_teams, n_teams, W), np.nan)
    for wk in range(W):
        act = np.where(active[:, wk])[0]
        if len(act) < 2:
            continue
        M = S[act, wk][:, None] - S[act, wk][None, :]
        Pw = 1.0 / (1.0 + np.exp(-M / K))
        np.fill_diagonal(Pw, 0.0)
        for row, i in enumerate(act):
            P_all[i, act, wk] = Pw[row]
            e[i, wk] = Pw[row].sum() / (len(act) - 1)
    delta = e - wins  # (N, W)
    F_margin = np.nansum(delta, axis=1)

    # ---- sanity: exact zero-mean per week when every team played ----
    for wk in range(W):
        if active[:, wk].all():
            assert np.allclose(f_count[wk].mean(), 0.0, atol=1e-12), \
                f"count method not centered in week {wk + 1}"
            assert np.allclose(delta[:, wk].mean(), 0.0, atol=1e-12), \
                f"margin method not centered in week {wk + 1}"

    stand = {clean_name(s["team"]): s for s in d["standings"]}
    print(f"K (logistic scale) = {K:.2f} points | weeks 1-{W} | {n_teams} teams")
    print(f"{'Team':<28} {'Rec':<6} {'IdealCnt':>8} {'F_count':>8} "
          f"{'ExpWins':>8} {'F_margin':>9}")
    for i, nm in enumerate(names):
        rec = stand.get(nm, {}).get("record", "?")
        print(f"{nm:<28} {rec:<6} {np.nansum(Wstar[:, i]):>8.0f} "
              f"{F_count[i]:>8.2f} {np.nansum(e[i, :] * (active.sum(axis=0) - 1)):>8.1f} "
              f"{F_margin[i]:>9.2f}")
    print("\nrank by F_count (neg=fraud):",
          [names[i] for i in np.argsort(F_count)])
    print("rank by F_margin (neg=fraud):",
          [names[i] for i in np.argsort(F_margin)])
    print("corr(F_count, F_margin) =",
          round(float(np.corrcoef(F_count, F_margin)[0, 1]), 4))

    json.dump({
        "names": names, "K": K, "N": n_teams, "N_weeks": W,
        "S": S.tolist(), "Wres": Wres.tolist(),
        "F_count": F_count.tolist(), "F_margin": F_margin.tolist(),
        "f_count_weekly": f_count.T.tolist(), "delta_weekly": delta.T.tolist(),
        "e_weekly": e.tolist(), "Wstar_weekly": Wstar.tolist(),
        "P_all": P_all.tolist(),
        "standings": {nm: stand.get(nm, {}).get("record", "?") for nm in names},
        "owners": {nm: (stand.get(nm, {}).get("owner") or {}) for nm in names},
        "opp": {f"{i},{w}": opp[(i, w)] for i in range(n_teams) for w in range(W)
                if (i, w) in opp},
    }, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
