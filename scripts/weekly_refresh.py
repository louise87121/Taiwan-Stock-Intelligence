from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def latest_friday(today: date | None = None) -> date:
    current = today or date.today()
    days_since_friday = (current.weekday() - 4) % 7
    return current - timedelta(days=days_since_friday)


def main() -> int:
    end_date = latest_friday().isoformat()
    command = [
        sys.executable,
        "-m",
        "src.main",
        "run-all",
        "--end-date",
        end_date,
    ]
    print(f"Running weekly refresh through {end_date}", flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        print(f"Weekly refresh failed with exit code {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
