import os
import time

import gridstatus
import pandas as pd
from gridstatus.ercot_60d_utils import (
    process_sced_esr,
    process_sced_gen,
    process_sced_load,
)

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

def get_as_prices(api, date, cache="data"):
    """ERCOT's real-time ancillary-service prices (MCPC, $/MW per hour) for every SCED run on date.

    One row per SCED run and product (ECRS, NSPIN, REGDN, REGUP, RRS). Source: product NP6-332-CD in the
    public API archive. ERCOT's website keeps these only about a week, and gridstatus's list of API
    endpoints predates them, so the archive is read directly. Cached as data/mcpc_{date}.pkl.
    """
    path = os.path.join(cache, f"mcpc_{date}.pkl")
    if os.path.exists(path):
        return one_price_per_run(pd.read_pickle(path))  # caches written before the duplicate fix may hold them

    # Each run's prices are posted seconds after the run, so postings from midnight to the next
    # midnight cover the day (calendar dates, so a 23- or 25-hour DST day is handled)
    tz = "US/Central"
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # The text after the product ID is a placeholder: the archive lookup reads only the ID
    df = retry(api.get_historical_data, "/np6-332-cd/x", pd.Timestamp(date, tz=tz), pd.Timestamp(nxt, tz=tz))

    # Time-zone-aware run times (handles the repeated fall-back hour), keep only this date's runs
    df = lambda_timestamps(df)
    df = df[df["SCED Timestamp"].dt.strftime("%Y-%m-%d") == date]
    # The archive changed format during 2026: late-July files have one price column, MCPC; by mid-September
    # it is split into CappedMCPC and UncappedMCPC. A single MCPC is used as the price compared with ERCOT;
    # whether it was capped isn't stated, so mcpc_uncapped is left empty for those days.
    if "CappedMCPC" in df.columns:
        capped, uncapped = df["CappedMCPC"], df["UncappedMCPC"]
    elif "MCPC" in df.columns:
        capped, uncapped = df["MCPC"], float("nan")
    else:
        raise ValueError(f"no MCPC column in the NP6-332-CD archive for {date}: columns {list(df.columns)}")
    out = df[["SCED Timestamp", "ASType"]].assign(
        mcpc=pd.to_numeric(capped, errors="coerce"),
        mcpc_uncapped=pd.to_numeric(uncapped, errors="coerce"),
    ).reset_index(drop=True)
    out = one_price_per_run(out)

    os.makedirs(cache, exist_ok=True)
    out.to_pickle(path)
    return out


def one_price_per_run(mcpc):
    """At most one row per (SCED run, product).

    The archive can list a run twice: 2026-08-27 has five runs (10:35-11:00) posted twice, every copy without a
    price. Identical copies collapse to one; where one copy has a price and the other doesn't, the priced one is
    kept. Two different prices for one run and product would mean a correction whose order the archive doesn't
    give, so that raises instead of guessing.
    """
    key = ["SCED Timestamp", "ASType"]
    out = mcpc.drop_duplicates()                                   # identical copies (NaN counts as equal)
    out = out.sort_values("mcpc", na_position="last", kind="stable")
    out = out[~(out.duplicated(key, keep=False) & out["mcpc"].isna())]  # a copy without a price, beside a priced one
    clash = out[out.duplicated(key, keep=False)]
    if not clash.empty:
        runs = sorted(clash["SCED Timestamp"].astype(str).unique())
        raise ValueError(f"different MCPCs posted for the same run and product: {runs[:5]}")
    return out.sort_values(key, kind="stable").reset_index(drop=True)


def get_as_demand_curves(api, date, cache="data"):
    """ERCOT's ancillary-service demand curves used by SCED on date: one curve per hour and product.

    Each curve is a list of points (MW, $/MW per hour). Source: product NP4-212-CD in the public API
    archive (website keeps about a month). Curves seen so far post around 04:55 on the delivery day;
    if a curve is posted more than once, the latest posting is kept. Cached as data/asdc_{date}.pkl.
    """
    path = os.path.join(cache, f"asdc_{date}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    # Look at postings from the day before through the day itself, in case some arrive a day early
    tz = "US/Central"
    prev = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = retry(api.get_historical_data, "/np4-212-cd/x", pd.Timestamp(prev, tz=tz), pd.Timestamp(nxt, tz=tz),
               add_post_datetime=True)

    # Keep this delivery date only, and the latest posting of each (hour, product) curve
    df = df[pd.to_datetime(df["DeliveryDate"]).dt.strftime("%Y-%m-%d") == date].copy()
    df["postDatetime"] = pd.to_datetime(df["postDatetime"])
    key = ["HourEnding", "RepeatedHourFlag", "ASType"]
    df = df[df["postDatetime"] == df.groupby(key)["postDatetime"].transform("max")]

    # One row per curve; points sorted by MW
    df = df.sort_values(key + ["DemandCurvePoint"])
    curves = (
        df.groupby(key)
        .apply(lambda g: list(zip(g["Quantity"].astype(float), g["Price"].astype(float), strict=True)),
               include_groups=False)
        .rename("curve")
        .reset_index()
    )

    os.makedirs(cache, exist_ok=True)
    curves.to_pickle(path)
    return curves

def get_sced_disclosure(date, ercot=None):
    """60-day SCED disclosure for date, downloaded once: resource tables processed, AS offers raw.

    gridstatus's processing (process=True) splits each AS offer into per-product curves and loses which
    block a quantity came from. So download raw (process=False) and apply gridstatus's own processing to
    the generator, battery (ESR) and Load Resource tables only. Same keys as process=True:
    sced_gen_resource, sced_esr, sced_load_resource (processed); sced_resource_as_offers (raw, block
    columns QUANTITY_MWn and PRICEn_<product> kept). A table missing on that date is None.
    """
    ercot = ercot or gridstatus.Ercot()
    raw = ercot.get_60_day_sced_disclosure(date, process=False)

    def proc(key, f):
        """Apply gridstatus's processing function f to table key, if the table exists."""
        return f(raw[key]) if raw.get(key) is not None else None

    return {
        "sced_gen_resource": proc("sced_gen_resource", process_sced_gen),
        "sced_esr": proc("sced_esr", process_sced_esr),
        "sced_load_resource": proc("sced_load_resource", process_sced_load),
        "sced_resource_as_offers": raw.get("sced_resource_as_offers"),  # raw on purpose
    }