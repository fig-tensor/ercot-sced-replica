import pandas as pd
from pytest import approx

from scedrep.build import build_interval


def test_build_keeps_only_online_dispatchable():
    gen = pd.DataFrame(
        {
            "SCED Timestamp": ["t1", "t1", "t1"],
            "Resource Name": ["A", "B", "C"],
            "Telemetered Resource Status": ["ON", "OFF", "ON"],
            "LDL": [10, 0, 50],
            "HDL": [100, 80, 50],
            "Base Point": [60, 0, 50],
            "SCED2 Offer Curve": [[[0, 20], [100, 30]], [[0, 5], [80, 5]], None],
        }
    )
    units, load = build_interval(gen, "t1")

    assert set(units) == {"A"}
    assert load == approx(60)
