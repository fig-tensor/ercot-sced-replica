import pandas as pd
from pytest import approx

from scedrep.validate import compare_as, run_day

TS = pd.Timestamp("2026-09-21 00:00:20", tz="US/Central")
PRODUCTS = ["REGUP", "REGDN", "RRS", "ECRS", "NSPIN"]


def test_run_day_with_as():
    # Same market as the interior-price case: energy A $10 (marginal, load 50), B $20.
    # RegUp: A 30 MW at $5, B 40 MW at $10, demand $100 for 50 MW -> A 30, B 20, MCPC = $10.
    gen = pd.DataFrame({
        "SCED Timestamp": [TS, TS], "Resource Name": ["A", "B"], "Telemetered Resource Status": ["ON", "ON"],
        "SCED2 Offer Curve": [[(0, 10), (100, 10)], [(0, 20), (100, 20)]],
        "LDL": [0.0, 0.0], "HDL": [100.0, 100.0], "HSL": [200.0, 100.0], "LSL": [0.0, 0.0],
        "Base Point": [50.0, 0.0],
    })
    blocks = pd.DataFrame({"SCED Timestamp": [TS, TS], "Resource Name": ["A", "B"], "Offer Type": ["ONRES"] * 2,
                           "block": [1, 1], "mw": [30.0, 40.0], "product": ["REGUP"] * 2, "price": [5.0, 10.0]})
    # ERCOT's awards: A 30, B 20 (what the model should reproduce), all other products 0
    awards = pd.DataFrame({"SCED Timestamp": [TS, TS], "Resource Name": ["A", "B"],
                           **{k: [0.0, 0.0] for k in PRODUCTS + ["RRSPFR", "RRSUFR", "RRSFFR"]}})
    awards["REGUP"] = [30.0, 20.0]
    asdc = pd.DataFrame([{"HourEnding": "01:00", "RepeatedHourFlag": "N", "ASType": k,
                          "curve": [(0, 100), (50 if k == "REGUP" else 1, 100)]} for k in PRODUCTS])

    res = run_day(gen, as_inputs=(blocks, awards, asdc))

    row = res.iloc[0]
    assert row["ok"]
    assert row["solver"] in {"piqp", "piqp-strict", "highs"}   # which solver produced the run is recorded
    assert row["model_lambda"] == approx(10, abs=1e-3)
    assert row["mcpc_REGUP"] == approx(10, abs=1e-3)
    assert row["cleared_REGUP"] == approx(50, abs=1e-3)
    assert row["award_tot_REGUP"] == approx(50)
    assert row["award_err_REGUP"] == approx(0, abs=1e-3)   # model awards match ERCOT's


def test_compare_as():
    res = pd.DataFrame({"SCED Timestamp": [TS], "mcpc_REGUP": [10.0], "mcpc_RRS": [2.0]})
    # ERCOT's prices, timestamp in another time-zone object: the join is on the instant
    mcpc = pd.DataFrame({"SCED Timestamp": [TS.tz_convert("UTC")] * 2, "ASType": ["REGUP", "RRS"],
                         "mcpc": [9.5, 2.0]})
    out = compare_as(res, mcpc).set_index("ASType")
    assert out.loc["REGUP", "err"] == approx(0.5)
    assert out.loc["RRS", "err"] == approx(0.0)