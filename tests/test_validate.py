import pandas as pd
from pytest import approx

from scedrep.validate import compare_day


def test_compare_day():
    t1 = pd.Timestamp("2026-09-08 00:00:20", tz="US/Central")
    t2 = pd.Timestamp("2026-09-08 00:05:20", tz="US/Central")

    # model results: one row per SCED run (what run_day returns)
    res = pd.DataFrame({"SCED Timestamp": [t1, t2], "model_lambda": [30.0, 40.0]})
    # ERCOT's published lambda, timestamps as text the way the API sends them
    lam = pd.DataFrame(
        {
            "SCEDTimestamp": ["2026-09-08T00:00:20", "2026-09-08T00:05:20"],
            "UncappedSystemLambda": [31.0, 45.0],
        }
    )
    # one binding constraint, at t2 only
    sp = pd.DataFrame({"SCED Timestamp": [t2], "Shadow Price": [12.5]})

    cmp = compare_day(res, lam, sp)

    assert list(cmp["err"]) == approx([-1.0, -5.0])
    assert list(cmp["congested"]) == [False, True]
