import argparse

import pandas as pd


def parse_args(argv=None):
    """Command line: scedrep DATE [END] [--as]. Returns (list of dates, with_as)."""
    p = argparse.ArgumentParser(prog="scedrep", description="Re-clear ERCOT SCED runs and compare with ERCOT.")
    p.add_argument("date", help="first date, YYYY-MM-DD (60-day disclosure: at least about 61 days ago)")
    p.add_argument("end", nargs="?", help="last date, YYYY-MM-DD (inclusive); default: DATE only")
    p.add_argument("--as", dest="with_as", action="store_true",
                   help="also run the model with ancillary services co-optimized (results/runs_as)")
    a = p.parse_args(argv)
    dates = pd.date_range(a.date, a.end or a.date).strftime("%Y-%m-%d").tolist()
    if not dates:
        p.error(f"END {a.end} is before DATE {a.date}")
    return dates, a.with_as


def main(argv=None) -> None:
    from scedrep.pipeline import run_date

    dates, with_as = parse_args(argv)
    if len(dates) == 1:  # one date: let any error surface as is (the daily run should fail loudly)
        run_date(dates[0], with_as=with_as)
        return
    failed = []
    for d in dates:  # a range: report a failed date and go on; a re-run skips the dates already done
        try:
            run_date(d, with_as=with_as)
        except Exception as e:  # noqa: BLE001
            failed.append(d)
            print(f"{d}: FAILED - {type(e).__name__}: {str(e)[:200]}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(dates)} dates failed: {', '.join(failed)}")
