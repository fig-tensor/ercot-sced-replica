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
    # 2026 run: the PFR rows are in (no PFR offered, so the floor is short), so a pricing run without them
    # is solved too; here it prices the same as the dispatch run
    assert row["pfr_short"] > 0
    assert row["model_lambda_pricing"] == approx(10, abs=1e-3)
    assert row["mcpcpr_REGUP"] == approx(10, abs=1e-3)


def test_compare_as():
    res = pd.DataFrame({"SCED Timestamp": [TS], "mcpc_REGUP": [10.0], "mcpc_RRS": [2.0]})
    # ERCOT's prices, timestamp in another time-zone object: the join is on the instant
    mcpc = pd.DataFrame({"SCED Timestamp": [TS.tz_convert("UTC")] * 2, "ASType": ["REGUP", "RRS"],
                         "mcpc": [9.5, 2.0]})
    out = compare_as(res, mcpc).set_index("ASType")
    assert out.loc["REGUP", "err"] == approx(0.5)
    assert out.loc["RRS", "err"] == approx(0.0)
    # the pricing run's columns, by prefix
    pr = compare_as(res.rename(columns={"mcpc_REGUP": "mcpcpr_REGUP", "mcpc_RRS": "mcpcpr_RRS"}), mcpc,
                    prefix="mcpcpr_").set_index("ASType")
    assert pr.loc["REGUP", "err"] == approx(0.5)

def test_as_price_report_and_lambda_report():
    from scedrep.validate import as_price_report, lambda_report
    t0 = pd.Timestamp("2026-08-31 19:00:20", tz="US/Central")
    t1 = pd.Timestamp("2026-09-01 10:00:20", tz="US/Central")
    a = pd.DataFrame({"SCED Timestamp": [t0, t1], "ok": [True, True], "err": [0.5, -2.0],
                      **{f"err_mcpc_{k}": [0.1, -0.3] for k in PRODUCTS},
                      **{f"award_tot_{k}": [10.0, 10.0] for k in PRODUCTS},
                      **{f"award_err_{k}": [1.0, 3.0] for k in PRODUCTS}})
    r = as_price_report(a)
    assert r.loc["RRS", "median |err|"] == approx(0.2)
    assert r.loc["RRS", "awards misplaced"] == approx(0.2)
    # energy only: same runs, both congested; read back from CSV, timestamps are strings
    b = pd.DataFrame({"SCED Timestamp": [str(t0), str(t1)], "ok": [True, True], "err": [3.0, -4.0],
                      "congested": [True, True]})
    lr = lambda_report(a, b)
    assert list(lr.index) == ["2026-08", "2026-09"]
    assert lr.loc["2026-08", "congested |err| energy"] == approx(3.0)
    assert lr.loc["2026-09", "congested |err| AS"] == approx(2.0)
    assert lr.loc["2026-08", "evening err AS"] == approx(0.5)
    assert pd.isna(lr.loc["2026-09", "evening err AS"])   # 10:00 is not evening
