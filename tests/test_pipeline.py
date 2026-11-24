import pandas as pd
from pytest import approx

from scedrep.pipeline import TZ, evening_median


def test_evening_median_uses_hours_18_to_21():
    # six hourly runs, 17:00 to 22:00 Central; only 18, 19, 20, 21 count
    ts = pd.date_range("2026-09-21 17:00", periods=6, freq="h", tz=TZ)
    cmp = pd.DataFrame({"SCED Timestamp": ts, "err": [100.0, 1.0, 2.0, 3.0, 4.0, 100.0]})
    assert evening_median(cmp) == approx(2.5)  # median of 1, 2, 3, 4

    # the same timestamps read back from CSV as strings give the same answer
    cmp["SCED Timestamp"] = cmp["SCED Timestamp"].astype(str)
    assert evening_median(cmp) == approx(2.5)