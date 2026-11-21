import os

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.fetch import get_lambda
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

    # lam = ERCOT system lambda for every SCED run on this date
    api = ErcotAPI()
    lam = get_lambda(api, date)

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
