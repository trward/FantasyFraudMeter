# FantasyFraudMeter

A fraud coefficient for head-to-head fantasy football: **how much of your
record is skill and how much is schedule luck.**

Every week, the meter compares what you *actually* did (one head-to-head
matchup) against what a hypothetical round-robin would have given you
(everyone plays everyone that week). If you posted the second-highest score
in the league but happened to play the highest scorer, you went 0-1 that week
while "beating" 10 of 11 other teams on score. That is measurable bad luck,
and this repo measures it.

**Fraud Meter 3.0** (this repo) improves on the classic count-based meter
("how many teams would I have beaten?") by weighting each hypothetical
matchup by the *margin* of the win, so a 2-point squeak and a 40-point blowout
are not treated the same.

- **Negative coefficient** = fraudulent: you won more than your scores say you deserved.
- **Positive coefficient** = unlucky: your scores say you deserved more wins than you got.
- **Zero** = the schedule neither helped nor hurt. (The meter is exactly
  zero-mean across the league every week, by construction.)

## Pipeline

```
pull_scores.py  ->  league_data.json   (ESPN: standings + weekly box scores)
compute_fraud.py -> fraud_data.json    (fraud coefficients, both variants)
plot_fraud.py    -> fraud_meter_3.png  (the meter + weekly trajectories)
```

## Setup

```bash
pip install -r requirements.txt
```

You need ESPN session cookies for your league. On fantasy.espn.com open
DevTools > Application > Cookies and copy the values of `espn_s2` and `swid`
into a file, one per line:

```
<espn_s2 value>
<swid value>
```

Default location is `~/.hermes/secrets/ff_cookies.txt` or the `$FF_COOKIES`
environment variable. Cookies expire; re-copy them when the pull fails with
`STALE_COOKIES`.

## Usage

```bash
# 1. Pull the league (defaults: league 2139522332, year 2026)
python pull_scores.py --league-id 2139522332 --year 2026 --out league_data.json

# 2. Compute the coefficients (default: weeks 1..current_week-1,
#    K = median absolute pairwise margin)
python compute_fraud.py --data league_data.json --out fraud_data.json

# 3. Plot
python plot_fraud.py --data fraud_data.json --out fraud_meter_3.png \
    --report-dir report/          # optional: light-themed report figures
```

## Method

Let `s_i(w)` be team `i`'s total score in week `w`, `n` the number of teams
that played that week, and `w_i(w) ∈ {0, 1}` the actual head-to-head result
(0.5 for a tie).

**Count-based (Fraud Meter 2.0).** The fraction of teams `i` would have
beaten in a hypothetical round-robin:

```
ideal_i(w) = #{j != i : s_i(w) > s_j(w)} / (n - 1)
delta_i(w) = ideal_i(w) - w_i(w)
```

**Margin-weighted (Fraud Meter 3.0).** Replace the binary "would have
beaten" with a logistic win probability in the margin:

```
p_ij(w)    = 1 / (1 + exp(-(s_i(w) - s_j(w)) / K))
ideal_i(w) = mean_{j != i} p_ij(w)
delta_i(w) = ideal_i(w) - w_i(w)
```

`K` is the scale parameter in points; the default is the median absolute
pairwise margin over the season (a team leading by `K` points is expected to
win 75% of such hypothetical matchups). Override with `--k`.

The season fraud coefficient is `F_i = sum_w delta_i(w)`.

**Why it is centered.** For every ordered pair `(i, j)`, `p_ij + p_ji = 1`,
so the sum of `ideal_i(w)` over all teams is exactly `(n-1)/2 = 0.5` per
team on average, matching the average actual win rate of 0.5. The meter
therefore sums to zero across the league in every week: nobody's unluckiness
is an illusion, it is someone else's fraud.

**Properties.**

- Bounded: `delta_i(w) ∈ [-1, 1]`, so `F_i ∈ [-W, +W]` over `W` weeks.
- A bye week (null score) excludes the team from that week's computation.
- The two variants agree closely (r ≈ 0.93 in the 2026 season through week
  9) but the margin-weighted version separates teams that the count-based
  version ties: winning by 1 point across the board is worth less than
  winning by 20.

## Output

`fraud_data.json` contains, per team: actual record, the count-based ideal
record (out of `11 * W` hypothetical games), the margin-weighted expected
wins, the weekly `delta` matrix, and both season coefficients.

`plot_fraud.py` produces:

- `fraud_meter_3.png` — the meter: a lollipop chart with four zones
  (`!!FRAUD ZONE!!`, `Sus`, `Chillin`, `Unlucky asf`) plus the cumulative
  week-by-week trajectory per team.
- With `--report-dir`: light-themed versions plus a count-vs-margin
  comparison scatter and a weekly delta heatmap, ready for a report.

Zone bounds default to `-1, 0, +1` (cumulative units); tune with
`--zone-bounds` for your league and season length.

## Disclaimer

This is a joke metric with real math. The coefficient measures schedule luck
in a 9-week sample of an 11-game round-robin; it is not a verdict on anyone's
integrity, their lineup decisions, or their ability to start a backup RB
over a stud. Science's most accurate representation of integrity and
honesty, amen.
