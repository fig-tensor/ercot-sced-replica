import pandas as pd
from pytest import approx

from scedrep.validate import compare_day, run_day


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

def test_run_day_with_network():
    # One SCED run, two units: A cheap ($20), B expensive ($50); ERCOT dispatched A=60, B=40
    ts = pd.Timestamp("2026-09-08 00:00:15", tz="US/Central")
    gen = pd.DataFrame({
        "SCED Timestamp": [ts, ts],
        "Resource Name": ["A", "B"],
        "Telemetered Resource Status": ["ON", "ON"],
        "SCED2 Offer Curve": [[(0, 20), (150, 20)], [(0, 50), (100, 50)]],
        "LDL": [0, 0],
        "HDL": [150, 100],
        "Base Point": [60.0, 40.0],
    })

    # A line carries A's output and binds at 60 MW in ERCOT's solution
    node = pd.Series({"A": "N1", "B": "N2"})
    SF = pd.DataFrame({"N1": [1.0], "N2": [0.0]}, index=["L1 | K1"])
    sp = pd.DataFrame({
        "SCED Timestamp": [ts],
        "Constraint Name": ["L1"],
        "Contingency Name": ["K1"],
        "Shadow Price": [30.0],
        "Limit": [60.0],
        "Value": [60.0],
        "Max Shadow Price": [1000.0],
    })

    # Without the network, A takes all the load and sets lambda = 20
    assert run_day(gen)["model_lambda"].iloc[0] == approx(20, abs=1e-3)

    # With the network, A is held at 60, B is marginal, lambda = 50
    res = run_day(gen, net=(node, SF, sp))
    assert res["model_lambda"].iloc[0] == approx(50, abs=1e-3)
    assert res["n_network"].iloc[0] == 1
