import os

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.validate import compare_day, run_day


def run_date(date, out="results/daily.csv"):
    # skip dates already in the summary file
    if os.path.exists(out) and date in pd.read_csv(out)["date"].astype(str).values:
        print(f"{date} already in {out}, skipping")
        return None

    # gen = 60-day SCED generator table for this date
    gen = gridstatus.Ercot().get_60_day_sced_disclosure(date, process=True)[
        "sced_gen_resource"
    ]

    # lam = ERCOT system lambda from date to date + 1 day
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    api = ErcotAPI()
    lam = api.hit_ercot_api(
        "/np6-322-cd/sced_system_lambda",
        SCEDTimestampFrom=f"{date}T00:00:00",
        SCEDTimestampTo=f"{nxt}T00:00:00",
    )

    # fallback to archive data when live endpoint returns nothing
    if len(lam) == 0:
        tz = "US/Central"
        lam = api.get_historical_data(
            "/np6-322-cd/sced_system_lambda",
            pd.Timestamp(date, tz=tz),
            pd.Timestamp(f"{nxt} 01:00", tz=tz),
        ).rename(
            columns={
                "SCEDTimeStamp": "SCEDTimestamp",
                "SystemLambda": "UncappedSystemLambda",
            }
        )

    # use capped lambda where ERCOT's uncapped value is missing (older days)
    if "CappedSystemLambda" in lam.columns:
        lam["UncappedSystemLambda"] = pd.to_numeric(
            lam["UncappedSystemLambda"], errors="coerce"
        ).fillna(lam["CappedSystemLambda"])

    # sp = SCED shadow prices for this date
    sp = api.get_shadow_prices_sced(date)

    # solve every SCED run, then compare to ERCOT
    res = run_day(gen)
    cmp = compare_day(res, lam, sp)

    # one-row summary for this date
    summary = pd.DataFrame(
        [
            {
                "date": date,
                "runs": len(cmp),
                "solved": int(cmp["ok"].sum()),
                "med_err_uncongested": cmp.loc[~cmp["congested"], "err"].median(),
                "med_err_congested": cmp.loc[cmp["congested"], "err"].median(),
            }
        ]
    )

    # append summary; write the header only if the file is new
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary.to_csv(out, mode="a", header=not os.path.exists(out), index=False)

    # per-run detail, one file per day (for the regression work)
    os.makedirs("results/runs", exist_ok=True)
    cmp.to_csv(f"results/runs/{date}.csv", index=False)

    return cmp
