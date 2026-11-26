import os

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.asbuild import PRODUCTS, QUALIFICATION, as_awards, as_offer_blocks
from scedrep.fetch import (
    get_as_demand_curves,
    get_as_prices,
    get_congestion,
    get_lambda,
    get_sced_disclosure,
    retry,
)
from scedrep.network import fit_shift_factors, unit_nodes
from scedrep.validate import compare_as, compare_day, run_day

WINDOW = 14  # days of history for the shift-factor fit
TZ = "US/Central"
AS_DIR = "results/runs_as"  # per-run detail of the AS model, one file per day


def fit_network(api, ercot, date, gen):
    """Unit -> node map and shift factors fitted on the WINDOW days before date.

    gen: any table with a "Resource Name" column; pass generators and batteries together
    so both get a node.
    """
    # Training days: date - WINDOW ... date - 1 (cached in data/ after the first download)
    train = pd.date_range(end=pd.Timestamp(date) - pd.Timedelta(days=1), periods=WINDOW).strftime("%Y-%m-%d")
    days = [get_congestion(api, d) for d in train]
    SF = fit_shift_factors([y for y, _ in days], [m for _, m in days])

    # Each dispatchable resource's node, from ERCOT's public mapping files
    rn = ercot.get_resource_node_to_unit(date)
    ccp = ercot.get_ccp_resource_names(date)
    node = unit_nodes(gen["Resource Name"].unique(), rn, ccp)
    return node, SF


def evening_median(cmp):
    """Median error over the evening net-peak hours 18-21 (18:00-21:59 Central)."""
    # utc=True then convert: works for tz-aware timestamps and for strings read back from CSV
    hour = pd.to_datetime(cmp["SCED Timestamp"], utc=True).dt.tz_convert(TZ).dt.hour
    return cmp.loc[hour.between(18, 21), "err"].median()


def as_table(res, lam, sp, mcpc):
    """Per-run table of the AS model: lambda error and, per product, ERCOT's price and the model's error.

    res: run_day output with AS (as_inputs given). Prices compared with ERCOT come from the floor-free solve
    (dispatch.solve_with_pricing) where there is one, else from the dispatch solve:
      err      = lambda (floor-free solve) - ERCOT's UncappedSystemLambda
      err_dispatch = the dispatch solve's lambda error (compare_day's err)
      mcpc_<k>, ercot_mcpc_<k>, err_mcpc_<k> = model MCPC, ERCOT's, model - ERCOT ($/MW-h)
    """
    cmp = compare_day(res, lam, sp).rename(columns={"err": "err_dispatch"})
    pricing = "model_lambda_pricing" in cmp.columns
    lam_used = cmp["model_lambda_pricing"].fillna(cmp["model_lambda"]) if pricing else cmp["model_lambda"]
    cmp["err"] = lam_used - cmp["UncappedSystemLambda"]

    # MCPC: floor-free prices where present (columns mcpcpr_<k>), else the dispatch solve's (mcpc_<k>)
    prefix = "mcpcpr_" if any(c.startswith("mcpcpr_") for c in res.columns) else "mcpc_"
    long = compare_as(res[res["ok"]], mcpc, prefix=prefix)
    for k in PRODUCTS:
        d = long[long["ASType"] == k].set_index("SCED Timestamp")
        cmp[f"mcpc_{k}"] = cmp["SCED Timestamp"].map(d["model"])
        cmp[f"ercot_mcpc_{k}"] = cmp["SCED Timestamp"].map(d["ercot"])
        cmp[f"err_mcpc_{k}"] = cmp["SCED Timestamp"].map(d["err"])
    return cmp


def as_summary(t):
    """Daily summary columns of the AS model, from as_table output."""
    ok = t[t["ok"]]
    cong = ok["congested"].astype(bool)
    row = {
        "solved_as": int(ok.shape[0]),
        "med_err_uncongested_as": ok.loc[~cong, "err"].median(),
        "med_err_congested_as": ok.loc[cong, "err"].median(),
        "med_err_evening_as": evening_median(ok),
    }
    for k in PRODUCTS:  # median |MCPC error| per product
        row[f"med_abs_err_mcpc_{k}"] = ok[f"err_mcpc_{k}"].abs().median()
    return row


def _download(date):
    """Everything one day needs: the 60-day disclosure (one download, AS offers kept raw), ERCOT's system lambda
    and shadow prices, and the API client."""
    api = ErcotAPI()
    return {
        "api": api,
        "ercot": gridstatus.Ercot(),
        "disc": get_sced_disclosure(date),
        "lam": retry(get_lambda, api, date),
        "sp": retry(api.get_shadow_prices_sced, date),
    }


def _solve_energy(date, d, node, SF):
    """The three energy-only models (unchanged, so daily.csv history stays comparable):
    cmp - no network, batteries fixed at base points; net - network, batteries fixed; both - network, batteries
    dispatchable."""
    gen, esr, lam, sp = d["disc"]["sced_gen_resource"], d["disc"]["sced_esr"], d["lam"], d["sp"]
    cmp = compare_day(run_day(gen), lam, sp)
    net = compare_day(run_day(gen, net=(node, SF, sp)), lam, sp)
    both = compare_day(run_day(gen, net=(node, SF, sp), esr=esr), lam, sp)
    return cmp, net, both


def _solve_as(date, d, node, SF):
    """The AS model: network, dispatchable batteries, energy and five AS products co-optimized, ERCOT's
    qualification durations for batteries. Returns as_table output."""
    disc, api = d["disc"], d["api"]
    gen, esr = disc["sced_gen_resource"], disc["sced_esr"]
    blocks = as_offer_blocks(disc["sced_resource_as_offers"])
    awards = as_awards(gen, esr, disc["sced_load_resource"])
    asdc = get_as_demand_curves(api, date)
    res = run_day(gen, net=(node, SF, d["sp"]), esr=esr, as_inputs=(blocks, awards, asdc), durations=QUALIFICATION)
    return as_table(res, d["lam"], d["sp"], get_as_prices(api, date))


def run_date(date, out="results/daily.csv", with_as=False):
    """Re-clear every SCED run of date and record how it compares with ERCOT.

    Energy-only models always (results/runs, runs_net, runs_both); with_as adds the AS model (results/runs_as).
    One summary row per date in out. A date already in out is skipped, except that with_as fills in the AS
    model for a date that has only the energy models.
    """
    summary = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=["date"])
    have = date in summary["date"].astype(str).values
    need_as = with_as and not os.path.exists(f"{AS_DIR}/{date}.csv")
    if have and not need_as:
        print(f"{date} already in {out}, skipping")
        return None

    d = _download(date)
    disc = d["disc"]
    node, SF = fit_network(d["api"], d["ercot"], date, pd.concat([disc["sced_gen_resource"], disc["sced_esr"]]))
    row = {}
    if not have:
        cmp, net, both = _solve_energy(date, d, node, SF)
        row = {
            "runs": len(cmp),
            "solved": int(cmp["ok"].sum()),
            "med_err_uncongested": cmp.loc[~cmp["congested"], "err"].median(),
            "med_err_congested": cmp.loc[cmp["congested"], "err"].median(),
            "solved_net": int(net["ok"].sum()),
            "med_err_congested_net": net.loc[net["congested"], "err"].median(),
            "solved_both": int(both["ok"].sum()),
            "med_err_uncongested_both": both.loc[~both["congested"], "err"].median(),
            "med_err_congested_both": both.loc[both["congested"], "err"].median(),
            "med_err_evening_both": evening_median(both),
        }
        for folder, df in [("results/runs", cmp), ("results/runs_net", net), ("results/runs_both", both)]:
            os.makedirs(folder, exist_ok=True)
            df.to_csv(f"{folder}/{date}.csv", index=False)
    if need_as:
        t = _solve_as(date, d, node, SF)
        row.update(as_summary(t))
        os.makedirs(AS_DIR, exist_ok=True)
        t.to_csv(f"{AS_DIR}/{date}.csv", index=False)

    # one row per date: update the date's row if present, else append; rewritten whole, so older rows get any new
    # columns (blank)
    summary = merge_row(summary, date, row)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    summary.to_csv(out, index=False)
    return row


def merge_row(summary, date, row):
    """summary with date's row set to row's values: existing columns of that row are kept unless row has them;
    a new date is appended. Rows stay in date order."""
    summary = summary.copy()
    summary["date"] = summary["date"].astype(str)
    new = pd.DataFrame([{"date": date, **row}])
    if date in summary["date"].values:
        for c in new.columns:
            if c not in summary.columns:
                summary[c] = float("nan")
            summary.loc[summary["date"] == date, c] = new[c].iloc[0]
    else:
        summary = pd.concat([summary, new], ignore_index=True)
    return summary.sort_values("date", ignore_index=True)
