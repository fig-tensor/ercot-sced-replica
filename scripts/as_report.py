"""Summarise the AS model's saved results: the README's ancillary-service tables, from results/ alone.

Reads results/runs_as/DATE.csv (written by `scedrep DATE --as`) and, for the energy-only comparison,
results/runs_both/DATE.csv. No downloads.
  uv run python scripts/as_report.py                         # every day in results/runs_as
  uv run python scripts/as_report.py 2026-07-27 2026-09-25   # a date range
"""

import glob
import os
import sys

import pandas as pd

from scedrep.validate import as_price_report, lambda_report


def load(folder, start=None, end=None):
    """All per-run files in folder with start <= date <= end, stacked."""
    parts = []
    for f in sorted(glob.glob(f"{folder}/*.csv")):
        d = os.path.basename(f)[:-4]
        if (start is None or d >= start) and (end is None or d <= end):
            parts.append(pd.read_csv(f).assign(date=d))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(start=None, end=None):
    pd.set_option("display.width", 200)
    a = load("results/runs_as", start, end)
    if a.empty:
        print("no AS results in results/runs_as for that range: run `scedrep DATE [END] --as` first")
        return
    print(f"{a['date'].nunique()} days, {len(a)} runs, {int(a['ok'].sum())} solved with AS")
    print("\nAS prices, $/MW-h:")
    print(as_price_report(a).round(3).to_string())
    b = load("results/runs_both", start, end)
    if not b.empty:
        print("\nlambda, $/MWh (energy only vs with AS, runs both solved):")
        print(lambda_report(a, b).round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
