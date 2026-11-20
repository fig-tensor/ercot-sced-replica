import os

import gridstatus
import pandas as pd
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.validate import compare_day, run_day


def run_date(date, out="results/daily.csv"):
    # gen = 60-day SCED generator table for this date (gridstatus, process=True)
    gen = gridstatus.Ercot().get_60_day_sced_disclosure(date, process=True)[
        "sced_gen_resource"
    ]
    # lam = ERCOT system lambda from date to date + 1 day (ErcotAPI().hit_ercot_api)
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    api = ErcotAPI()
    lam = api.hit_ercot_api(
        "/np6-322-cd/sced_system_lambda",
        SCEDTimestampFrom=f"{date}T00:00:00",
        SCEDTimestampTo=f"{nxt}T00:00:00",
    )
    # sp = SCED shadow prices for this date (ErcotAPI().get_shadow_prices_sced)
    sp = api.get_shadow_prices_sced(date)
    # res = run_day(gen); cmp = compare_day(res, lam, sp)
    res = run_day(gen)
    cmp = compare_day(res, lam, sp)
    # summary = one-row table: date, runs, solved, median err (uncongested), median err (congested)
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
    # append summary to out; write the header only if the file is new
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    # return cmp
    return cmp
