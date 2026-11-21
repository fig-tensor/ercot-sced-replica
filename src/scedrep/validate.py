import pandas as pd
from pyomo.contrib.solver.common.util import NoDualsError, NoOptimalSolutionError

from scedrep.build import build_interval
from scedrep.dispatch import solve_interval


def run_day(gen, penalty=5000):
    rows = []
    # for each distinct SCED timestamp, in time order:
    for ts in sorted(gen["SCED Timestamp"].unique()):
        print(f"{ts}", flush=True)
        #     build units and load for that timestamp
        units, load = build_interval(gen, ts)
        #     try to solve; record {"SCED Timestamp": ts, "model_lambda": ..., "shortage": ..., "ok": True}
        try:
            r = solve_interval(units, load, penalty=penalty)
            rows.append(
                {
                    "SCED Timestamp": ts,
                    "model_lambda": r["lambda"],
                    "shortage": r["shortage"],
                    "ok": True,
                }
            )
        #     if solving raises an error, record the timestamp with "ok": False instead
        except (NoOptimalSolutionError, NoDualsError) as e:
            rows.append(
                {
                    "SCED Timestamp": ts,
                    "model_lambda": None,
                    "shortage": None,
                    "ok": False,
                    "error": repr(e),
                }
            )
    # return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def compare_day(res, lam, sp):
    lam = lam.copy()
    # ERCOT lambda: parse SCEDTimestamp into tz-aware "SCED Timestamp" (US/Central)
    lam["SCED Timestamp"] = pd.to_datetime(lam["SCEDTimestamp"]).dt.tz_localize(
        "US/Central"
    )
    # join res to lam on "SCED Timestamp" (keep every model run)
    cmp = res.merge(
        lam[["SCED Timestamp", "UncappedSystemLambda"]], on="SCED Timestamp", how="left"
    )
    # err = model_lambda - UncappedSystemLambda
    cmp["err"] = cmp["model_lambda"] - cmp["UncappedSystemLambda"]
    # congested = timestamp has any constraint with Shadow Price > 0 in sp
    cong = sp.loc[sp["Shadow Price"] > 0, "SCED Timestamp"].unique()
    cmp["congested"] = cmp["SCED Timestamp"].isin(cong)
    # return the joined table
    return cmp
