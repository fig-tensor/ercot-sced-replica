import os

import pandas as pd

from scedrep.fetch import retry

TZ = "US/Central"

# ERCOT's region and zone names as they appear in the column headers
WIND = ["PANHANDLE", "COASTAL", "SOUTH", "WEST", "NORTH"]
SOLAR = ["CenterWest", "NorthWest", "FarWest", "FarEast", "SouthEast", "CenterEast"]
LOAD = ["Coast", "East", "Far West", "North", "North Central", "South Central", "Southern", "West"]
OUTAGE = ["South", "North", "West", "Houston"]


def snapshot(df, target, cutoff_hour=9):
    """The forecast for day `target` as it stood at cutoff_hour:00 Central the day before.

    df:     a multi-vintage forecast table (ERCOT re-publishes every hour), with
            time-zone-aware "Interval Start" and "Publish Time" columns.
    target: the day being forecast, "YYYY-MM-DD".
    Returns the target day's rows from the last report published at or before the cutoff.
    """
    # Target day [start, end) and the cutoff, all in Central time
    start = pd.Timestamp(target).tz_localize(TZ)
    end = (pd.Timestamp(target) + pd.Timedelta(days=1)).tz_localize(TZ)
    cut = (pd.Timestamp(target) - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour)).tz_localize(TZ)

    # Latest report known at the cutoff; nothing published later may be used (no leakage)
    pub = df.loc[df["Publish Time"] <= cut, "Publish Time"].max()
    if pd.isna(pub):
        raise ValueError(f"no report published by {cut} for {target}")

    # That report's rows for the target day
    rows = df[(df["Publish Time"] == pub) & (df["Interval Start"] >= start) & (df["Interval Start"] < end)]
    return rows.sort_values("Interval Start").reset_index(drop=True)


def build_features(wind, solar, load, outage, target, cutoff_hour=9):
    """One row per hour of `target`: ERCOT's day-ahead forecasts as known at the cutoff."""
    w = snapshot(wind, target, cutoff_hour).set_index("Interval Start")
    s = snapshot(solar, target, cutoff_hour).set_index("Interval Start")
    # Load: only the forecast model ERCOT itself was using
    ld = snapshot(load[load["In Use Flag"]], target, cutoff_hour).set_index("Interval Start")
    o = snapshot(outage, target, cutoff_hour).set_index("Interval Start")

    X = pd.DataFrame(index=w.index)
    for r in WIND:
        X[f"wind {r}"] = w[f"STWPF {r}"]                              # ERCOT's main wind forecast
        X[f"wind spread {r}"] = w[f"STWPF {r}"] - w[f"WGRPP {r}"]     # its uncertainty band
    for r in SOLAR:
        X[f"solar {r}"] = s[f"STPPF {r}"]
        X[f"solar spread {r}"] = s[f"STPPF {r}"] - s[f"PVGRPP {r}"]
    for z in LOAD:
        X[f"load {z}"] = ld[z]
    for z in OUTAGE:
        X[f"outage {z}"] = o[f"Total Resource MW Zone {z}"]          # MW of generation out

    # Calendar: hour of day, day of week, month (local time)
    X["hour"] = X.index.hour
    X["weekday"] = X.index.dayofweek
    X["month"] = X.index.month
    return X


def day_features(api, target, cache="data", cutoff_hour=9):
    """Feature table for one target day, downloaded once and cached in data/."""
    path = os.path.join(cache, f"features_{target}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    # Every report published on the day before the target; snapshot then keeps the one at the cutoff
    d1 = (pd.Timestamp(target) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    wind = retry(api.get_wind_actual_and_forecast_by_geographical_region_hourly, d1)
    solar = retry(api.get_solar_actual_and_forecast_by_geographical_region_hourly, d1)
    load = retry(api.get_load_forecast_by_model, d1)
    outage = retry(api.get_hourly_resource_outage_capacity, d1)

    X = build_features(wind, solar, load, outage, target, cutoff_hour)
    os.makedirs(cache, exist_ok=True)
    X.to_pickle(path)
    return X

def hourly_binding(sp, hours):
    """1 if constraint c bound in at least one SCED run during hour t, else 0.

    sp:    one day's SCED shadow prices (gridstatus get_shadow_prices_sced).
    hours: the hours to cover (the feature table's index), time-zone aware.
    Returns hours x constraints ("element | contingency"), entries 0 or 1.
    """
    # Binding constraints only, keyed the same way as the shift-factor work
    b = sp[sp["Shadow Price"] > 0]
    key = b["Constraint Name"] + " | " + b["Contingency Name"]

    # Hour each SCED run falls in; flooring in UTC is safe because Central's offsets are whole hours
    hour = b["SCED Timestamp"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(TZ)

    # Count binding runs per (hour, constraint), then cap at 1: "bound at least once"
    y = pd.crosstab(hour, key).clip(upper=1)
    y.index.name, y.columns.name = None, None

    # Every hour of the day, with 0 where nothing bound
    return y.reindex(hours, fill_value=0)


def day_targets(api, target, hours, cache="data"):
    """Binding table for one target day, downloaded once and cached in data/."""
    path = os.path.join(cache, f"targets_{target}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)

    sp = retry(api.get_shadow_prices_sced, target)
    y = hourly_binding(sp, hours)
    os.makedirs(cache, exist_ok=True)
    y.to_pickle(path)
    return y