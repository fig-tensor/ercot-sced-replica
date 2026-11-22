import pandas as pd

TZ = "US/Central"


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