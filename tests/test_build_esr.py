import pandas as pd
from pytest import approx

from scedrep.build import build_interval


def test_build_interval_with_batteries():
    ts = pd.Timestamp("2026-09-21 00:00:20", tz="US/Central")
    cols = ["SCED Timestamp", "Resource Name", "Telemetered Resource Status", "SCED2 Offer Curve", "LDL", "HDL", "Base Point"]
    gen = pd.DataFrame([[ts, "A", "ON", [(0, 20), (100, 20)], 0.0, 100.0, 60.0]], columns=cols)
    # E is charging at 20 MW; F is out of service and must be left out
    esr = pd.DataFrame([
        [ts, "E", "ON", [(-50, 10), (0, 10), (0, 40), (50, 40)], -50.0, 50.0, -20.0],
        [ts, "F", "OUT", [(0, -250)], 0.0, 0.0, 0.0],
    ], columns=cols)

    units, load = build_interval(gen, ts, esr)

    assert set(units) == {"A", "E"}
    assert units["E"]["ldl"] == -50.0          # can charge down to -50 MW
    assert load == approx(60.0 - 20.0)         # the battery's charging counts in the balance

    # Without the battery table, only the generator
    units, load = build_interval(gen, ts)
    assert set(units) == {"A"}
    assert load == approx(60.0)