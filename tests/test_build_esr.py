import pandas as pd
from pytest import approx

from scedrep.build import build_interval


def test_build_interval_with_batteries():
    ts = pd.Timestamp("2026-09-21 00:00:20", tz="US/Central")
    cols = ["SCED Timestamp", "Resource Name", "Telemetered Resource Status", "SCED2 Offer Curve", "LDL", "HDL", "Base Point"]
    gen = pd.DataFrame([[ts, "A", "ON", [(0, 20), (100, 20)], 0.0, 100.0, 60.0]], columns=cols)
    # SOC columns: E is half full, so its SOC cap (room / interval = 50 MWh x 12 = 600 MW) doesn't bind.
    esr = pd.DataFrame([
        [ts, "E", "ON", [(-50, 10), (0, 10), (0, 40), (50, 40)], -50.0, 50.0, -20.0, 50.0, 0.0, 100.0],
        [ts, "F", "OUT", [(0, -250)], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], columns=cols + ["SOC", "Min SOC", "Max SOC"])

    units, load = build_interval(gen, ts, esr)

    assert set(units) == {"A", "E"}
    assert units["E"]["ldl"] == -50.0          # can charge down to -50 MW
    assert load == approx(60.0 - 20.0)         # the battery's charging counts in the balance

    # Without the battery table, only the generator
    units, load = build_interval(gen, ts)
    assert set(units) == {"A"}
    assert load == approx(60.0)

def test_full_battery_cannot_charge():
    ts = pd.Timestamp("2026-09-21 04:45", tz="US/Central")
    gen = pd.DataFrame({"SCED Timestamp": [ts], "Resource Name": ["G"], "Telemetered Resource Status": ["ON"],
                        "SCED2 Offer Curve": [[(0, 10), (100, 10)]], "LDL": [0.0], "HDL": [100.0], "Base Point": [50.0]})
    esr = pd.DataFrame({"SCED Timestamp": [ts], "Resource Name": ["B"], "Telemetered Resource Status": ["ON"],
                        "SCED2 Offer Curve": [[(-50, 30), (0, 30)]], "LDL": [-50.0], "HDL": [50.0], "Base Point": [0.0],
                        "SOC": [100.0], "Min SOC": [0.0], "Max SOC": [100.0]})
    units, _ = build_interval(gen, ts, esr)
    assert units["B"]["ldl"] == 0          # full: no room to charge
    assert units["B"]["hdl"] == 50         # 100 MWh stored, so HDL (50) still binds