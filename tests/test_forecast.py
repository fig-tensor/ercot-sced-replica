import pandas as pd
import pytest

import scedrep.forecast as fc
from scedrep.forecast import (
    LOAD,
    OUTAGE,
    SOLAR,
    WIND,
    build_features,
    hourly_binding,
    snapshot,
)


def ts(s):
    return pd.Timestamp(s, tz="US/Central")


def test_snapshot_uses_last_report_before_cutoff():
    # Three reports, each forecasting hours 0 and 1 of Sept 15 (plus one Sept 14 row)
    pubs = ["2026-09-13 08:00", "2026-09-14 08:55", "2026-09-14 10:00"]
    rows = []
    for k, p in enumerate(pubs):
        for h in ["2026-09-14 23:00", "2026-09-15 00:00", "2026-09-15 01:00"]:
            rows.append({"Publish Time": ts(p), "Interval Start": ts(h), "STWPF WEST": 100 * (k + 1)})
    df = pd.DataFrame(rows)

    out = snapshot(df, "2026-09-15")

    # The 08:55 report on Sept 14 is the last one before 09:00; the 10:00 one is too late
    assert (out["Publish Time"] == ts("2026-09-14 08:55")).all()
    assert out["STWPF WEST"].tolist() == [200, 200]

    # Only the target day's hours, in order
    assert out["Interval Start"].tolist() == [ts("2026-09-15 00:00"), ts("2026-09-15 01:00")]


def test_snapshot_without_a_report_before_cutoff():
    # The only report is published after the cutoff, so there is nothing legal to use
    df = pd.DataFrame({
        "Publish Time": [ts("2026-09-14 10:00")],
        "Interval Start": [ts("2026-09-15 00:00")],
        "STWPF WEST": [1.0],
    })
    with pytest.raises(ValueError):
        snapshot(df, "2026-09-15")

def report(pub, cols, value):
    # One report published at `pub`, covering hours 0 and 1 of Sept 15, every column = value
    hours = [ts("2026-09-15 00:00"), ts("2026-09-15 01:00")]
    df = pd.DataFrame({"Publish Time": [ts(pub)] * 2, "Interval Start": hours})
    for c in cols:
        df[c] = float(value)
    return df


def test_build_features():
    # Each input has one report before the cutoff (value 1) and one after it (value 99)
    def two(cols):
        return pd.concat([report("2026-09-14 08:00", cols, 1), report("2026-09-14 12:00", cols, 99)])

    wind = two([f"{k} {r}" for r in WIND for k in ("STWPF", "WGRPP")])
    solar = two([f"{k} {r}" for r in SOLAR for k in ("STPPF", "PVGRPP")])
    load = two(LOAD)
    load["In Use Flag"] = True
    outage = two([f"Total Resource MW Zone {z}" for z in OUTAGE])

    X = build_features(wind, solar, load, outage, "2026-09-15")

    # Two hours, and only the pre-cutoff values (1, never 99)
    assert len(X) == 2
    assert X["wind WEST"].tolist() == [1.0, 1.0]
    assert X["wind spread WEST"].tolist() == [0.0, 0.0]   # STWPF - WGRPP = 1 - 1
    assert X["load Far West"].tolist() == [1.0, 1.0]
    assert X["outage Houston"].tolist() == [1.0, 1.0]
    assert X["hour"].tolist() == [0, 1]

def test_hourly_binding():
    # A binds twice in hour 0; C binds once in hour 1; B has a zero shadow price (not binding)
    sp = pd.DataFrame({
        "SCED Timestamp": [ts("2026-09-15 00:05"), ts("2026-09-15 00:10"), ts("2026-09-15 01:05"), ts("2026-09-15 01:30")],
        "Constraint Name": ["A", "A", "C", "B"],
        "Contingency Name": ["K", "K", "K", "K"],
        "Shadow Price": [12.0, 8.0, 3.0, 0.0],
    })
    hours = pd.DatetimeIndex([ts("2026-09-15 00:00"), ts("2026-09-15 01:00"), ts("2026-09-15 02:00")])

    y = hourly_binding(sp, hours)

    assert sorted(y.columns) == ["A | K", "C | K"]     # B never bound
    assert y["A | K"].tolist() == [1, 0, 0]            # two runs in hour 0 still count as 1
    assert y["C | K"].tolist() == [0, 1, 0]
    assert list(y.index) == list(hours)                # hour 2: no binding, still present

def test_dataset_stacks_days_and_skips_bad_ones(monkeypatch):
    # Day 1: constraint A binds in hour 0. Day 2: B binds in hour 1. Day 3: no report (skipped).
    def fake_features(api, d, cache):
        if d == "2026-09-17":
            raise ValueError("no report")
        hours = pd.DatetimeIndex([ts(f"{d} 00:00"), ts(f"{d} 01:00")])
        return pd.DataFrame({"wind WEST": [1.0, 2.0]}, index=hours)

    def fake_targets(api, d, hours, cache):
        name = "A | K" if d == "2026-09-15" else "B | K"
        col = [1, 0] if d == "2026-09-15" else [0, 1]
        return pd.DataFrame({name: col}, index=hours)

    monkeypatch.setattr(fc, "day_features", fake_features)
    monkeypatch.setattr(fc, "day_targets", fake_targets)

    X, Y = fc.dataset(None, "2026-09-15", "2026-09-17")

    assert len(X) == 4                              # two good days x two hours; day 3 skipped
    assert list(Y.index) == list(X.index)
    assert Y["A | K"].tolist() == [1, 0, 0, 0]      # A absent on day 2: filled with 0
    assert Y["B | K"].tolist() == [0, 0, 0, 1]

def test_build_features_fills_missing_outage_hour():
    # The outage report lacks hour 1 (as on the fall-back DST day): it takes hour 0's value
    wind = report("2026-09-14 08:00", [f"{k} {r}" for r in WIND for k in ("STWPF", "WGRPP")], 1)
    solar = report("2026-09-14 08:00", [f"{k} {r}" for r in SOLAR for k in ("STPPF", "PVGRPP")], 1)
    load = report("2026-09-14 08:00", LOAD, 1)
    load["In Use Flag"] = True
    outage = report("2026-09-14 08:00", [f"Total Resource MW Zone {z}" for z in OUTAGE], 7).iloc[:1]

    X = build_features(wind, solar, load, outage, "2026-09-15")

    assert X["outage West"].tolist() == [7.0, 7.0]