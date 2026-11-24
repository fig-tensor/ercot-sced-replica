import os

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.fetch import get_congestion, get_lambda, retry
from scedrep.network import fit_shift_factors, unit_nodes
from scedrep.validate import compare_day, run_day

WINDOW = 14  # days of history for the shift-factor fit
TZ = "US/Central"


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


def run_date(date, out="results/daily.csv"):
    # skip dates already in the summary file
    if os.path.exists(out) and date in pd.read_csv(out)["date"].astype(str).values:
        print(f"{date} already in {out}, skipping")
        return None

    # 60-day SCED disclosure for this date: generators and batteries (ESRs) come in one download
    ercot = gridstatus.Ercot()
    disc = ercot.get_60_day_sced_disclosure(date, process=True)
    gen = disc["sced_gen_resource"]
    esr = disc["sced_esr"]

    # ERCOT's system lambda and shadow prices for this date
    api = ErcotAPI()
    lam = retry(get_lambda, api, date)
    sp = retry(api.get_shadow_prices_sced, date)

    # solve every SCED run three times:
    #   cmp  - no network, batteries fixed at base points (original baseline)
    #   net  - network, batteries fixed
    #   both - network, batteries dispatchable
    # The first two stay unchanged so daily.csv history stays comparable.
    cmp = compare_day(run_day(gen), lam, sp)
    node, SF = fit_network(api, ercot, date, pd.concat([gen, esr]))  # nodes for units and batteries
    net = compare_day(run_day(gen, net=(node, SF, sp)), lam, sp)
    both = compare_day(run_day(gen, net=(node, SF, sp), esr=esr), lam, sp)

    # one-row summary for this date
    summary = pd.DataFrame([{
        "date": date,
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
    }])

    # add to the summary file; rewrite it whole so older rows get the new columns (blank)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        summary = pd.concat([pd.read_csv(out), summary], ignore_index=True)
    summary.to_csv(out, index=False)

    # per-run detail, one file per day, for each model
    for folder, df in [("results/runs", cmp), ("results/runs_net", net), ("results/runs_both", both)]:
        os.makedirs(folder, exist_ok=True)
        df.to_csv(f"{folder}/{date}.csv", index=False)

    return cmp