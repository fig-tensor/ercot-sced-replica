import os

import pandas as pd
import pytest
from pytest import approx

import scedrep.pipeline as P
from scedrep import parse_args
from scedrep.pipeline import TZ, evening_median


def test_evening_median_uses_hours_18_to_21():
    # six hourly runs, 17:00 to 22:00 Central; only 18, 19, 20, 21 count
    ts = pd.date_range("2026-09-21 17:00", periods=6, freq="h", tz=TZ)
    cmp = pd.DataFrame({"SCED Timestamp": ts, "err": [100.0, 1.0, 2.0, 3.0, 4.0, 100.0]})
    assert evening_median(cmp) == approx(2.5)  # median of 1, 2, 3, 4

    # the same timestamps read back from CSV as strings give the same answer
    cmp["SCED Timestamp"] = cmp["SCED Timestamp"].astype(str)
    assert evening_median(cmp) == approx(2.5)

# ---------- AS model in the pipeline ----------


T0 = pd.Timestamp("2026-09-21 19:00:20", tz=TZ)
T1 = pd.Timestamp("2026-09-21 19:05:20", tz=TZ)


def lam_sp():
    """ERCOT's lambda ($30, $40) and shadow prices (T1 congested) for two runs, as the loaders return them."""
    lam = pd.DataFrame({"SCEDTimestamp": ["2026-09-21 19:00:20", "2026-09-21 19:05:20"],
                        "RepeatedHourFlag": ["N", "N"], "UncappedSystemLambda": [30.0, 40.0]})
    sp = pd.DataFrame({"SCED Timestamp": [T1], "Shadow Price": [12.0]})
    return lam, sp


def test_as_table_uses_floor_free_prices():
    # Two runs. Dispatch solve lambda 31/45; floor-free 30.5/41 -> err 0.5/1.0 (err_dispatch 1/5).
    # MCPC: floor-free RegUp 10 against ERCOT 9.5 -> +0.5; at T1 ERCOT has no price for RegUp -> NaN.
    res = pd.DataFrame({"SCED Timestamp": [T0, T1], "ok": [True, True], "model_lambda": [31.0, 45.0],
                        "model_lambda_pricing": [30.5, 41.0],
                        **{f"mcpc_{k}": [99.0, 99.0] for k in P.PRODUCTS},
                        **{f"mcpcpr_{k}": [10.0, 8.0] for k in P.PRODUCTS}})
    mcpc = pd.DataFrame({"SCED Timestamp": [T0] * 5 + [T1] * 4, "ASType": P.PRODUCTS + P.PRODUCTS[1:],
                         "mcpc": [9.5, 10, 10, 10, 10, 8, 8, 8, 7]})
    lam, sp = lam_sp()
    t = P.as_table(res, lam, sp, mcpc)
    assert list(t["err"]) == pytest.approx([0.5, 1.0])
    assert list(t["err_dispatch"]) == pytest.approx([1.0, 5.0])
    assert list(t["congested"]) == [False, True]
    assert t["err_mcpc_REGUP"].iloc[0] == pytest.approx(0.5)
    assert pd.isna(t["err_mcpc_REGUP"].iloc[1])
    assert t["err_mcpc_NSPIN"].iloc[1] == pytest.approx(1.0)
    assert t["ercot_mcpc_REGDN"].iloc[0] == pytest.approx(10)

    s = P.as_summary(t)
    assert s["solved_as"] == 2
    assert s["med_err_uncongested_as"] == pytest.approx(0.5)
    assert s["med_err_congested_as"] == pytest.approx(1.0)
    assert s["med_err_evening_as"] == pytest.approx(0.75)   # both runs are at 19:00
    assert s["med_abs_err_mcpc_REGUP"] == pytest.approx(0.5)


def test_as_table_without_floor_free_solve():
    # No PFR rows -> no floor-free columns: lambda and MCPC from the dispatch solve
    res = pd.DataFrame({"SCED Timestamp": [T0], "ok": [True], "model_lambda": [31.0],
                        **{f"mcpc_{k}": [10.0] for k in P.PRODUCTS}})
    mcpc = pd.DataFrame({"SCED Timestamp": [T0] * 5, "ASType": P.PRODUCTS, "mcpc": [9.0] * 5})
    lam, sp = lam_sp()
    t = P.as_table(res, lam, sp, mcpc)
    assert t["err"].iloc[0] == pytest.approx(1.0)
    assert t["err_mcpc_ECRS"].iloc[0] == pytest.approx(1.0)


def test_merge_row_updates_appends_and_sorts():
    s = pd.DataFrame({"date": ["2026-09-02", "2026-09-01"], "runs": [288, 288]})
    s = P.merge_row(s, "2026-09-02", {"solved_as": 287})          # fills a new column on an existing row
    assert list(s["date"]) == ["2026-09-01", "2026-09-02"]
    assert s.loc[s["date"] == "2026-09-02", "runs"].item() == 288  # kept
    assert s.loc[s["date"] == "2026-09-02", "solved_as"].item() == 287
    assert pd.isna(s.loc[s["date"] == "2026-09-01", "solved_as"].item())
    s = P.merge_row(s, "2026-08-31", {"runs": 288})                # appended, in date order
    assert list(s["date"]) == ["2026-08-31", "2026-09-01", "2026-09-02"]


def fake_pipeline(monkeypatch, calls):
    """run_date with the downloads and solves replaced by stubs that record what ran."""
    one = pd.DataFrame({"SCED Timestamp": [T0], "ok": [True], "congested": [False], "err": [0.1]})
    monkeypatch.setattr(P, "_download", lambda d: {"api": None, "ercot": None,
                                                   "disc": {"sced_gen_resource": pd.DataFrame({"Resource Name": []}),
                                                            "sced_esr": pd.DataFrame({"Resource Name": []})}})
    monkeypatch.setattr(P, "fit_network", lambda *a: (None, None))
    monkeypatch.setattr(P, "_solve_energy", lambda *a: calls.append("energy") or (one, one, one))
    monkeypatch.setattr(P, "as_summary", lambda t: {"solved_as": 1})
    monkeypatch.setattr(P, "_solve_as", lambda *a: calls.append("as") or one)


def test_run_date_skip_and_fill(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    fake_pipeline(monkeypatch, calls)
    out = "results/daily.csv"
    P.run_date("2026-09-01", out)                    # new date, energy only
    assert calls == ["energy"]
    P.run_date("2026-09-01", out)                    # already there: skipped
    assert calls == ["energy"]
    P.run_date("2026-09-01", out, with_as=True)      # AS filled in, energy not re-run
    assert calls == ["energy", "as"]
    assert os.path.exists(f"{P.AS_DIR}/2026-09-01.csv")
    P.run_date("2026-09-01", out, with_as=True)      # both there: skipped
    assert calls == ["energy", "as"]
    P.run_date("2026-09-02", out, with_as=True)      # new date with AS: both
    assert calls == ["energy", "as", "energy", "as"]
    s = pd.read_csv(out)
    assert list(s["date"]) == ["2026-09-01", "2026-09-02"] and list(s["solved_as"]) == [1, 1]


def test_parse_args():
    assert parse_args(["2026-09-01"]) == (["2026-09-01"], False)
    assert parse_args(["2026-09-01", "2026-09-03", "--as"]) == (["2026-09-01", "2026-09-02", "2026-09-03"], True)
    with pytest.raises(SystemExit):
        parse_args(["2026-09-03", "2026-09-01"])
