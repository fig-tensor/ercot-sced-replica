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
            "RepeatedHourFlag": [False, False],
            "UncappedSystemLambda": [31.0, 45.0],
        }
    )
    # one binding constraint, at t2 only
    sp = pd.DataFrame({"SCED Timestamp": [t2], "Shadow Price": [12.5]})

    cmp = compare_day(res, lam, sp)

    assert list(cmp["err"]) == approx([-1.0, -5.0])
    assert list(cmp["congested"]) == [False, True]


def test_compare_day_repeated_hour():
    first = pd.Timestamp("2026-11-01 01:00:23").tz_localize(
        "US/Central", ambiguous=True
    )
    second = pd.Timestamp("2026-11-01 01:00:23").tz_localize(
        "US/Central", ambiguous=False
    )

    res = pd.DataFrame(
        {"SCED Timestamp": [first, second], "model_lambda": [60.0, 50.0]}
    )
    lam = pd.DataFrame(
        {
            "SCEDTimestamp": ["2026-11-01T01:00:23", "2026-11-01T01:00:23"],
            "RepeatedHourFlag": [False, True],
            "UncappedSystemLambda": [59.98138, 49.33632],
        }
    )

    sp = pd.DataFrame({"SCED Timestamp": [], "Shadow Price": []})

    cmp = compare_day(res, lam, sp)

    assert list(cmp["err"]) == approx([60.0 - 59.98138, 50.0 - 49.33632])
