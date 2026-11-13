#!/usr/bin/env python3
"""Pull ESPN fantasy football league data (standings + weekly box scores).

Usage:
    python pull_scores.py [--league-id ID] [--year YEAR] [--cookies PATH]
                          [--max-week N] [--out FILE]

Cookies: a plain-text file with espn_s2 on line 1 and swid on line 2, copied
from fantasy.espn.com (DevTools > Application > Cookies). Defaults to the
$FF_COOKIES env var or ~/.hermes/secrets/ff_cookies.txt.

Output: a JSON file with league_name, current_week, standings[], and
weeks{week: {matchups[], teams{team_id: {name, score}}}}.
"""
import argparse
import json
import os
import sys

try:
    from espn_api.football import League
    from espn_api.requests.espn_requests import ESPNAccessDenied
except ImportError:
    print("espn_api not installed. Run: pip install espn_api", file=sys.stderr)
    sys.exit(3)


def load_cookies(path):
    try:
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) < 2:
            raise ValueError("need espn_s2 (line 1) and swid (line 2)")
        return lines[0], lines[1]
    except FileNotFoundError:
        print(f"STALE_COOKIES: missing {path}. Copy espn_s2/swid from "
              "fantasy.espn.com (DevTools > Application > Cookies) into this file.",
              file=sys.stderr)
        sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="Pull ESPN fantasy football league data")
    ap.add_argument("--league-id", type=int, default=2139522332)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--cookies",
                    default=os.environ.get("FF_COOKIES",
                                           os.path.expanduser("~/.hermes/secrets/ff_cookies.txt")))
    ap.add_argument("--max-week", type=int, default=17,
                    help="stop pulling after this week (default 17)")
    ap.add_argument("--out", default="league_data.json")
    args = ap.parse_args()

    espn_s2, swid = load_cookies(args.cookies)

    try:
        league = League(league_id=args.league_id, year=args.year,
                        espn_s2=espn_s2, swid=swid)
    except ESPNAccessDenied:
        print("STALE_COOKIES: ESPN rejected the session (espn_s2/swid expired).",
              file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        msg = str(e)
        if "403" in msg or "Forbidden" in msg:
            print(f"STALE_COOKIES: {msg}", file=sys.stderr)
            sys.exit(2)
        print(f"league fetch failed: {msg}", file=sys.stderr)
        sys.exit(1)

    out = {"current_week": None, "standings": [], "weeks": {}}
    try:
        out["league_name"] = league.settings.name
    except Exception as e:
        out["settings_error"] = str(e)
    try:
        out["current_week"] = league.current_week
    except Exception:
        pass

    team_names = {t.team_id: t.team_name for t in league.teams}
    team_abbrevs = {t.team_id: getattr(t, "abbrev", None) for t in league.teams}
    team_owners = {t.team_id: (t.owners[0] if t.owners else None) for t in league.teams}

    try:
        for team in league.standings():
            out["standings"].append({
                "team": team.team_name,
                "team_id": team.team_id,
                "owner": (team.owners[0] if team.owners else None),
                "rank": team.standing,
                "record": f"{team.wins}-{team.losses}-{team.ties}",
                "pf": team.points_for,
                "pa": team.points_against,
            })
        out["standings"].sort(key=lambda r: r["rank"] or 99)
    except Exception as e:
        out["standings_error"] = str(e)

    for week in range(1, args.max_week + 1):
        try:
            bss = league.box_scores(week=week)
        except Exception as e:
            if week > 10:
                break
            out[f"week{week}_error"] = f"{type(e).__name__}: {e}"
            continue
        if not bss:
            break
        matchups, teams = [], {}
        for bs in bss:
            for tid, score in ((bs.home_team, bs.home_score),
                               (bs.away_team, bs.away_score)):
                if tid is None:
                    continue
                teams[str(tid)] = {
                    "name": team_names.get(tid, str(tid)),
                    "abbrev": team_abbrevs.get(tid),
                    "owner": team_owners.get(tid),
                    "score": score,
                }
            matchups.append({
                "home": bs.home_team, "home_score": bs.home_score,
                "away": bs.away_team, "away_score": bs.away_score,
            })
        out["weeks"][str(week)] = {"matchups": matchups, "teams": teams}

    with open(args.out, "w") as f:
        json.dump(out, f, default=str)
    n_weeks = len(out["weeks"])
    print(f"saved {args.out}: {out.get('league_name')}, "
          f"current_week={out['current_week']}, {n_weeks} weeks of box scores, "
          f"{len(out['standings'])} teams")


if __name__ == "__main__":
    main()
