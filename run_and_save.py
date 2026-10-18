"""
Runs one full scan and writes it to results.json -- this is the only new
code in the scheduled architecture. cfl_engine.run_full_scan() itself is
completely unchanged from the validated Colab/backend version; this
script just calls it and saves the output instead of returning it from
an API or printing it to a notebook cell.
"""
import json
import os
import sys
from datetime import datetime, timezone

import cfl_engine


def completed_fingerprint(games, live_week):
    """Short hash of every FINAL score from weeks before the current one.
    It only changes when a week completes (or a past score is corrected),
    never while the current week is in progress."""
    import hashlib
    rows = sorted((g["gm"], g["hpf"], g["apf"]) for g in games
                  if g["wk"] < live_week and g["hpf"] is not None)
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def should_run():
    """Cheap gate (a couple of seconds, no solver): only run the heavy scan
    when a week has just been COMPLETED. The "current week" is the earliest
    week that still has an unplayed game, so it only changes once every game
    of the previous week has a final score. If it matches the week already
    saved in results.json, either that week is still in progress or we've
    already scanned this state -- nothing to do. A manual run from the
    Actions tab (FORCE_RUN=true) always runs."""
    if os.environ.get("FORCE_RUN", "").lower() == "true":
        print("Manual run -- skipping the week-complete check.")
        return True
    try:
        fixtures = cfl_engine.fetch_pims_fixtures(cfl_engine.SEASON_ID)
        games = cfl_engine.convert_fixtures_to_games(fixtures)
        _, live_week = cfl_engine.get_current_week_games_and_number(games)
        live_fp = completed_fingerprint(games, live_week)
    except ValueError:
        print("No unplayed regular-season games left -- season is complete. Nothing to do.")
        return False
    try:
        with open("results.json") as f:
            saved = json.load(f)
            saved_week, saved_fp = saved.get("current_week"), saved.get("completed_fingerprint")
    except (FileNotFoundError, ValueError):
        print("No saved results yet -- running.")
        return True
    if live_week == saved_week and live_fp == saved_fp:
        print(f"Week {live_week} is still in progress (or already scanned) -- skipping.")
        return False
    if live_week == saved_week:
        print("Saved results don't match the completed games -- re-running to correct them.")
        return True
    print(f"Week {saved_week} is complete; Week {live_week} is next -- running the scan.")
    return True


def main():
    if not should_run():
        return
    print("Starting scheduled scan...")
    try:
        result = cfl_engine.run_full_scan(send_email=False, verbose=True)
    except Exception as e:
        # Write a clearly-marked error file rather than silently leaving
        # the previous (possibly stale) results.json in place with no
        # indication anything went wrong.
        error_payload = {
            "error": str(e),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open("scan_error.json", "w") as f:
            json.dump(error_payload, f, indent=2)
        print(f"Scan failed: {e}", file=sys.stderr)
        sys.exit(1)

    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    # Fingerprint of the completed games this result was built from, so the
    # check above can tell whether the saved result is still valid.
    fixtures = cfl_engine.fetch_pims_fixtures(cfl_engine.SEASON_ID)
    games = cfl_engine.convert_fixtures_to_games(fixtures)
    result["completed_fingerprint"] = completed_fingerprint(games, result["current_week"])

    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scan complete. Week {result['current_week']}. "
          f"Cross-check {'OK' if result['cross_check_ok'] else 'MISMATCH -- investigate'}. "
          f"Wrote results.json.")


if __name__ == "__main__":
    main()
