"""
Runs one full scan and writes it to results.json -- this is the only new
code in the scheduled architecture. cfl_engine.run_full_scan() itself is
completely unchanged from the validated Colab/backend version; this
script just calls it and saves the output instead of returning it from
an API or printing it to a notebook cell.
"""
import json
import sys
from datetime import datetime, timezone

import cfl_engine


def main():
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

    with open("results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scan complete. Week {result['current_week']}. "
          f"Cross-check {'OK' if result['cross_check_ok'] else 'MISMATCH -- investigate'}. "
          f"Wrote results.json.")


if __name__ == "__main__":
    main()
