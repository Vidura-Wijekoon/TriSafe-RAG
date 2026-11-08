"""Case-study driver: Civil Documents (CS1).

Runs the offline reproduction backend on this domain slice of the
SL-PubServ-QA benchmark and writes results/summary.json plus per-query traces.
The neural-backend numbers for the same slice come from
results/neural/results.json (scripts/run_benchmark.py --backend neural).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from casestudies._common import print_table, run_case  # noqa: E402


def main() -> None:
    out = run_case(Path(__file__).parent)
    print_table("Civil Documents (CS1)", out)
    print(f"\nWrote {out['summary_path']}")


if __name__ == "__main__":
    main()
