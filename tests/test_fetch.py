import pandas as pd

from scedrep.fetch import get_congestion


class FakeAPI:
    """Stands in for ErcotAPI: one SCED run, one node, one binding constraint."""

    def __init__(self):
        self.calls = 0

    def hit_ercot_api(self, endpoint, **params):
        self.calls += 1
        return pd.DataFrame({
            "SCEDTimestamp": ["2026-09-08T00:00:15"],
            "RepeatedHourFlag": [False],
            "UncappedSystemLambda": [30.0],
        })

    def get_lmp_by_settlement_point(self, date):
        self.calls += 1
        ts = pd.Timestamp("2026-09-08 00:00:15", tz="US/Central")
        return pd.DataFrame({
            "SCED Timestamp": [ts],
            "Location": ["N1"],
            "Location Type": ["Resource Node"],
            "LMP": [26.0],
        })

    def get_shadow_prices_sced(self, date):
        self.calls += 1
        ts = pd.Timestamp("2026-09-08 00:00:15", tz="US/Central")
        return pd.DataFrame({
            "SCED Timestamp": [ts],
            "Constraint Name": ["L1"],
            "Contingency Name": ["K1"],
            "Shadow Price": [8.0],
        })


def test_get_congestion_builds_and_caches(tmp_path):
    api = FakeAPI()

    # First call downloads (3 API calls) and builds Y = LMP - lambda, M = shadow prices
    Y, M = get_congestion(api, "2026-09-08", cache=tmp_path)
    assert Y.iloc[0]["N1"] == -4.0
    assert M.iloc[0]["L1 | K1"] == 8.0
    assert api.calls == 3

    # Second call reads the cache: no new API calls
    get_congestion(api, "2026-09-08", cache=tmp_path)
    assert api.calls == 3