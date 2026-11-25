import pandas as pd
from pyomo.contrib.solver.common.util import NoDualsError, NoOptimalSolutionError

from scedrep.asbuild import PRODUCTS, build_as_interval
from scedrep.build import build_interval
from scedrep.dispatch import SolveError, solve_with_pricing
from scedrep.fetch import lambda_timestamps
from scedrep.network import build_network


def as_summary(out, units, awards, ts):
    """Per product: model MCPC, cleared MW, and award error vs ERCOT over the dispatched units.

    award_err_k = sum over units of |model award - ERCOT award|; award_tot_k = sum of ERCOT awards.
    """
    a = awards[(awards["SCED Timestamp"] == ts) & awards["Resource Name"].isin(units.keys())]
    ercot = a.set_index("Resource Name")
    row = {}
    for k in PRODUCTS:
        model = pd.Series({u: out["awards"].get(u, {}).get(k, 0.0) for u in units})
        actual = ercot[k].reindex(model.index).fillna(0.0) if k in ercot else model * 0
        row[f"mcpc_{k}"] = out["mcpc"].get(k)
        row[f"cleared_{k}"] = out["cleared"].get(k)
        row[f"award_err_{k}"] = float((model - actual).abs().sum())
        row[f"award_tot_{k}"] = float(actual.sum())
    return row


def run_day(gen, penalty=5000, net=None, esr=None, as_inputs=None, durations=None, max_runs=None):
    """Solve every SCED run in gen. net = (node, SF, sp) adds network constraints; esr makes batteries
    dispatchable; as_inputs = (blocks, awards, asdc) adds AS co-optimization (durations: battery duration
    rows, {product: hours}). max_runs: solve only the first N runs (for timing). With the RRS-PFR rows in, each
    run is solved twice (dispatch.solve_with_pricing): the pricing run's prices are the columns
    model_lambda_pricing and mcpcpr_<product>; pfr_short is the dispatch run's PFR shortfall (MW)."""
    rows = []
    for ts in sorted(gen["SCED Timestamp"].unique())[:max_runs]:
        print(f"{ts}", flush=True)
        units, load = build_interval(gen, ts, esr)
        row = {"SCED Timestamp": ts, "ok": False}
        asin = None
        if as_inputs is not None:
            blocks, awards, asdc = as_inputs
            asin = build_as_interval(blocks, awards, asdc, ts, units)

        # Without a network: one attempt. With one: all shift factors first, and if the
        # QP stalls, retry with tiny ones (|SF| < 0.01) cut, which stops the cycling
        cutoffs = [None] if net is None else [0.0, 0.01]
        for sf_min in cutoffs:
            network = []
            if sf_min is not None:
                node, SF, sp = net
                bp = pd.Series({u: units[u]["bp"] for u in units})
                network = build_network(bp, node, SF, sp[sp["SCED Timestamp"] == ts], sf_min)
            try:
                out = solve_with_pricing(units, load, penalty, network, asin=asin, durations=durations)
            except (NoOptimalSolutionError, NoDualsError, SolveError) as e:
                row["error"] = repr(e)
                continue
            row.update({
                "model_lambda": out["lambda"],
                "shortage": out["shortage"],
                "n_network": len(network),
                "violation": sum(out["violation"]),
                "sf_min": sf_min,
                "solver": out["solver"],  # which solver in the fallback chain produced this run
                "ok": True,
            })
            if asin is not None:
                row.update(as_summary(out, units, as_inputs[1], ts))
                row["pfr_short"] = out.get("pfr_short")
            if "pricing" in out:  # pricing run (no PFR rows): its prices beside the dispatch run's
                pr = out["pricing"]
                row["model_lambda_pricing"] = pr["lambda"]
                row.update({f"mcpcpr_{k}": v for k, v in pr["mcpc"].items()})
            break

        rows.append(row)
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


def compare_as(res, mcpc, prefix="mcpc_"):
    """Model AS prices vs ERCOT's: one row per (run, product) with model, ERCOT and err = model - ERCOT.

    res: run_day output with <prefix><product> columns (prefix "mcpc_": dispatch run; "mcpcpr_": pricing run);
    mcpc: fetch.get_as_prices output (SCED Timestamp, ASType, mcpc).
    """
    cols = [f"{prefix}{k}" for k in PRODUCTS if f"{prefix}{k}" in res.columns]
    long = res.melt(id_vars="SCED Timestamp", value_vars=cols, var_name="ASType", value_name="model")
    long["ASType"] = long["ASType"].str.removeprefix(prefix)
    # join on the instant (UTC), so time-zone objects from different sources still match
    long["t"] = pd.to_datetime(long["SCED Timestamp"], utc=True)
    m = mcpc.assign(t=pd.to_datetime(mcpc["SCED Timestamp"], utc=True))[["t", "ASType", "mcpc"]]
    out = long.merge(m.rename(columns={"mcpc": "ercot"}), on=["t", "ASType"], how="left").drop(columns="t")
    out["err"] = out["model"] - out["ercot"]
    return out