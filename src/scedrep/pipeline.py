import os

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.fetch import get_congestion, get_lambda, retry
from scedrep.network import fit_shift_factors, unit_nodes
from scedrep.validate import compare_day, run_day

WINDOW = 14  # days of history for the shift-factor fit


def fit_network(api, ercot, date, gen):
    """Unit -> node map and shift factors fitted on the WINDOW days before date."""
    # Training days: date - WINDOW ... date - 1 (cached in data/ after the first download)
    train = pd.date_range(end=pd.Timestamp(date) - pd.Timedelta(days=1), periods=WINDOW).strftime("%Y-%m-%d")
    days = [get_congestion(api, d) for d in train]
    SF = fit_shift_factors([y for y, _ in days], [m for _, m in days])

    # Each dispatchable resource's node, from ERCOT's public mapping files
    rn = ercot.get_resource_node_to_unit(date)
    ccp = ercot.get_ccp_resource_names(date)
    node = unit_nodes(gen["Resource Name"].unique(), rn, ccp)
    return node, SF


def run_date(date, out="results/daily.csv"):
    # skip dates already in the summary file
    if os.path.exists(out) and date in pd.read_csv(out)["date"].astype(str).values:
        print(f"{date} already in {out}, skipping")
        return None

    # gen = 60-day SCED generator table for this date
    ercot = gridstatus.Ercot()
    gen = ercot.get_60_day_sced_disclosure(date, process=True)["sced_gen_resource"]

    # ERCOT's system lambda and shadow prices for this date
    api = ErcotAPI()
    lam = retry(get_lambda, api, date)
    sp = retry(api.get_shadow_prices_sced, date)

    # solve every SCED run twice: without and with network constraints
    cmp = compare_day(run_day(gen), lam, sp)
    node, SF = fit_network(api, ercot, date, gen)
    net = compare_day(run_day(gen, net=(node, SF, sp)), lam, sp)

    # one-row summary for this date
    summary = pd.DataFrame([{
        "date": date,
        "runs": len(cmp),
        "solved": int(cmp["ok"].sum()),
        "med_err_uncongested": cmp.loc[~cmp["congested"], "err"].median(),
        "med_err_congested": cmp.loc[cmp["congested"], "err"].median(),
        "solved_net": int(net["ok"].sum()),
        "med_err_congested_net": net.loc[net["congested"], "err"].median(),
    }])

    # add to the summary file; rewrite it whole so older rows get the new columns (blank)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        summary = pd.concat([pd.read_csv(out), summary], ignore_index=True)
    summary.to_csv(out, index=False)

    # per-run detail, one file per day, for each model
    os.makedirs("results/runs", exist_ok=True)
    os.makedirs("results/runs_net", exist_ok=True)
    cmp.to_csv(f"results/runs/{date}.csv", index=False)
    net.to_csv(f"results/runs_net/{date}.csv", index=False)

    return cmp