import os
import time

import gridstatus
import pandas as pd

from scedrep.network import congestion_matrices


def retry(f, *args, **kwargs):
    """Call f(*args, **kwargs); if ERCOT rate-limits, wait 60 s and try again (up to 5 times)."""
    for _ in range(5):
        try:
            return f(*args, **kwargs)
        except RuntimeError:
            print("rate-limited, waiting 60 s", flush=True)
            time.sleep(60)
    return f(*args, **kwargs)


def get_lambda(api, date):
    """ERCOT system lambda for every SCED run on date (live API, archive fallback)."""
    # Live endpoint: every SCED run from date to date + 1 day
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lam = api.hit_ercot_api(
        "/np6-322-cd/sced_system_lambda",
        SCEDTimestampFrom=f"{date}T00:00:00",
        SCEDTimestampTo=f"{nxt}T00:00:00",
    )

    # Fall back to archive data when the live endpoint returns nothing
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

    # Use capped lambda where ERCOT's uncapped value is missing (older days)
    if "CappedSystemLambda" in lam.columns:
        lam["UncappedSystemLambda"] = pd.to_numeric(
            lam["UncappedSystemLambda"], errors="coerce"
        ).fillna(lam["CappedSystemLambda"])

    return lam


def lambda_timestamps(lam):
    """Add a time-zone-aware "SCED Timestamp" column to ERCOT's system-lambda table."""
    lam = lam.copy()
    # Fall-back DST: the repeated 1 a.m. hour is flagged True (live API) or "Y" (archive)
    flag = lam["RepeatedHourFlag"].astype(str).str.strip().str.upper().isin(["Y", "TRUE"])
    # ambiguous=~flag: first pass through 1 a.m. is daylight time, the repeat is standard time
    lam["SCED Timestamp"] = pd.to_datetime(lam["SCEDTimestamp"]).dt.tz_localize("US/Central", ambiguous=~flag)
    return lam


def get_congestion(api, date, cache="data", types=("Resource Node",), prefix="ym"):
    """One day's congestion matrices Y and M, downloaded once and cached as data/{prefix}_{date}.pkl.

    types: which settlement points go in Y (default resource nodes). For hubs and load zones use
    types=("Trading Hub", "Load Zone") with a different prefix, e.g. "hz", so the caches stay separate.
    """
    # Reuse the cached matrices if this day was built before
    path = os.path.join(cache, f"{prefix}_{date}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    # lam: lambda per SCED run, with time-zone-aware timestamps
    lam = lambda_timestamps(retry(get_lambda, api, date))
    lam["lam"] = pd.to_numeric(lam["UncappedSystemLambda"], errors="coerce")

    # LMPs at every settlement point, and SCED shadow prices
    lmp = retry(api.get_lmp_by_settlement_point, date)
    sp = retry(api.get_shadow_prices_sced, date)

    # Build the matrices and save them for next time
    Y, M = congestion_matrices(lmp, lam, sp, types)
    os.makedirs(cache, exist_ok=True)
    pd.to_pickle((Y, M), path)
    return Y, M

def save_unplanned_outages(publish_date, out_dir="results/outages"):
    """Save ERCOT's unplanned resource outage report published on `publish_date`, one CSV per report.

    ERCOT keeps only about 31 days of these on its website and they are not in the public API,
    so the daily run saves each one to build up history. Each report lists every unplanned
    outage open about three days before its publish date.
    """
    path = os.path.join(out_dir, f"{publish_date}.csv")
    if os.path.exists(path):
        return path
    df = gridstatus.Ercot().get_unplanned_resource_outages(pd.Timestamp(publish_date, tz="US/Central"))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return path