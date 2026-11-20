import pandas as pd
from pyomo.contrib.solver.common.util import NoDualsError, NoOptimalSolutionError

from scedrep.build import build_interval
from scedrep.dispatch import solve_interval


def run_day(gen, penalty=5000):
    rows = []
    # for each distinct SCED timestamp, in time order:
    for ts in sorted(gen["SCED Timestamp"].unique()):
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
