import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    # The outage report has no row for the repeated hour on the fall-back DST day; outages
    # change slowly, so a missing hour takes the previous hour's value
    out_cols = [f"outage {z}" for z in OUTAGE]
    X[out_cols] = X[out_cols].ffill().bfill()
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


def dataset(api, start, end, cache="data"):
    """Stacked features X and binding targets Y for every day from start to end (inclusive)."""
    Xs, ys = [], []
    for d in pd.date_range(start, end).strftime("%Y-%m-%d"):
        print(d, flush=True)
        try:
            X = day_features(api, d, cache)
            y = day_targets(api, d, X.index, cache)
        except (ValueError, KeyError) as e:
            # A missing report or a changed column skips the day rather than stopping the run
            print(f"  skipped {d}: {e!r}", flush=True)
            continue
        Xs.append(X)
        ys.append(y)

    X = pd.concat(Xs)
    # A constraint with no column on some day didn't bind that day: fill with 0
    Y = pd.concat(ys).reindex(X.index).fillna(0).astype(int)
    return X, Y

def shift_days(t, days):
    """Move a local timestamp by whole calendar days (midnight stays midnight across DST changes)."""
    return (t.tz_localize(None) + pd.Timedelta(days=days)).tz_localize(TZ)

def model_features(X):
    """Turn the raw feature table into model inputs: hour on a circle, weekend flag, no month."""
    F = X.drop(columns=["hour", "month", "weekday"])
    F["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)   # 23:00 sits next to 00:00
    F["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    F["weekend"] = (X["weekday"] >= 5).astype(int)
    return F

def loading(X, sens_c):
    """Predicted flow change on one constraint each hour: s_wind * total wind + s_solar * total solar.

    X:      raw feature table (regional wind and solar forecasts).
    sens_c: that constraint's type sensitivities (from network.type_sensitivity), e.g. {"WIND": 0.3, "PVGR": -0.1}.
    """
    wind = X[[f"wind {r}" for r in WIND]].sum(axis=1)       # ERCOT's systemwide wind forecast, MW
    solar = X[[f"solar {r}" for r in SOLAR]].sum(axis=1)    # systemwide solar forecast, MW
    return sens_c.get("WIND", 0.0) * wind + sens_c.get("PVGR", 0.0) * solar


def recent_rate(Y, days=14, lag=2):
    """Baseline: share of hours each constraint bound over `days` full days ending `lag` days before.

    With lag=2, day D uses days D-15 ... D-2: the last full days known at 9 AM on D-1.
    Returns a table shaped like Y (NaN where there isn't enough history yet).
    """
    day_sum = Y.resample("D").sum()
    day_hours = Y.iloc[:, 0].resample("D").count()
    rate = day_sum.rolling(days).sum().shift(lag).div(day_hours.rolling(days).sum().shift(lag), axis=0)
    out = rate.reindex(Y.index.floor("D"))
    out.index = Y.index
    return out


def walk_forward(X, Y, start, end, train_weeks=8, min_freq=0.05, C=0.01, sens=None):
    """Weekly walk-forward test of the binding classifier against two baselines.

    For each test week: train on the `train_weeks` weeks before it (ending one day before, since
    labels for D-1 aren't known at the cutoff), pick targets binding in >= min_freq of training
    hours, fit one L2-penalized logistic regression per target, predict the week.
    Returns one row per (hour, constraint) with the outcome y and three predictions:
    "recent" (14-day rate), "persist" (same hour two days earlier) and "model".
    sens: optional {test-week start: type sensitivities (constraints x types)}. If given, each
    constraint's model also gets its predicted loading (see `loading`) as a feature.
    """
    F = model_features(X)
    recent = recent_rate(Y)
    persist = Y.copy()
    persist.index = persist.index + pd.Timedelta(hours=48)
    persist = persist.reindex(Y.index)

    out = []
    for s in pd.date_range(start, end, freq="7D", tz=TZ):
        # Calendar-day arithmetic in local time, so weeks that cross a DST change keep every hour
        e = shift_days(s, 7)
        tr = (Y.index >= shift_days(s, -7 * train_weeks)) & (Y.index < shift_days(s, -1))
        te = (Y.index >= s) & (Y.index < e)
        if te.sum() == 0:
            continue
        freq = Y[tr].mean()
        for c in freq[freq >= min_freq].index:
            y_tr, y_te = Y.loc[tr, c], Y.loc[te, c]
            r = recent.loc[te, c].fillna(freq[c])
            Ftr, Fte = F[tr], F[te]
            if sens is not None:
                # This constraint's predicted loading; 0 if it had no shift-factor estimate that week
                s_c = sens[s].loc[c] if c in sens[s].index else pd.Series(dtype=float)
                f = loading(X, s_c)
                Ftr, Fte = Ftr.assign(loading=f[tr]), Fte.assign(loading=f[te])
            if y_tr.nunique() == 2:
                model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
                model.fit(Ftr, y_tr)
                p = model.predict_proba(Fte)[:, 1]
            else:
                p = r.to_numpy()   # nothing to learn from a constant label: fall back to the recent rate
            out.append(pd.DataFrame({
                "week": s, "constraint": c, "y": y_te.to_numpy(), "recent": r.to_numpy(),
                "persist": persist.loc[te, c].fillna(0).to_numpy(), "model": p,
            }, index=y_te.index))
    return pd.concat(out)


def score(R, methods=("recent", "persist", "model")):
    """Brier score, skill against the recent rate, and mean per-constraint AUC for each method."""
    ref = brier_score_loss(R["y"], R["recent"])
    rows = {}
    for m in methods:
        b = brier_score_loss(R["y"], R[m])
        aucs = [roc_auc_score(g["y"], g[m]) for _, g in R.groupby("constraint") if g["y"].nunique() == 2]
        rows[m] = {"brier": b, "skill": 1 - b / ref, "auc": float(np.mean(aucs))}
    return pd.DataFrame(rows).T