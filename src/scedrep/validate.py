import pandas as pd
from pyomo.contrib.solver.common.util import NoDualsError, NoOptimalSolutionError

from scedrep.build import build_interval
from scedrep.dispatch import solve_interval
from scedrep.fetch import lambda_timestamps
from scedrep.network import build_network


def run_day(gen, penalty=5000, net=None):
    """Solve every SCED run in gen. net = (node, SF, sp) adds network constraints."""
    rows = []
    for ts in sorted(gen["SCED Timestamp"].unique()):
        print(f"{ts}", flush=True)
        units, load = build_interval(gen, ts)

        # Network constraints for this run (none if net is None)
        network = []
        if net is not None:
            node, SF, sp = net
            bp = pd.Series({u: units[u]["bp"] for u in units})
            network = build_network(bp, node, SF, sp[sp["SCED Timestamp"] == ts])

        try:
            out = solve_interval(units, load, penalty, network)
            rows.append({
                "SCED Timestamp": ts,
                "model_lambda": out["lambda"],
                "shortage": out["shortage"],
                "n_network": len(network),
                "violation": sum(out["violation"]),
                "ok": True,
            })
        except (NoOptimalSolutionError, NoDualsError) as e:
            rows.append({"SCED Timestamp": ts, "ok": False, "error": repr(e)})
    return pd.DataFrame(rows)

def compare_day(res, lam, sp):
    lam = lambda_timestamps(lam)
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
