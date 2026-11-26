import pandas as pd

from scedrep.fetch import get_as_demand_curves, get_as_prices


class FakeArchive:
    """Stands in for ErcotAPI.get_historical_data: returns canned archive rows and counts calls."""

    def __init__(self, df):
        self.df, self.calls = df, 0

    def get_historical_data(self, endpoint, start, end, **kwargs):
        self.calls += 1
        return self.df


def test_get_as_prices_parses_filters_and_caches(tmp_path):
    raw = pd.DataFrame({
        "SCEDTimestamp": ["09/21/2026 12:25:21", "09/21/2026 12:25:21", "09/22/2026 00:00:15"],
        "RepeatedHourFlag": ["N", "N", "N"],
        "ASType": ["REGUP", "RRS", "REGUP"],
        "CappedMCPC": [1.60, 0.01, 9.0],
        "UncappedMCPC": [1.60, 0.01, 9.0],
    })
    api = FakeArchive(raw)

    out = get_as_prices(api, "2026-09-21", cache=tmp_path)
    # The next day's run is dropped; the two products on Sept 21 remain, with aware timestamps
    assert len(out) == 2
    assert out["SCED Timestamp"].iloc[0] == pd.Timestamp("2026-09-21 12:25:21", tz="US/Central")
    assert out.set_index("ASType").loc["REGUP", "mcpc"] == 1.60

    # Second call reads the cache
    get_as_prices(api, "2026-09-21", cache=tmp_path)
    assert api.calls == 1


def test_get_as_prices_older_single_column(tmp_path):
    # Late-July 2026 archive files have one MCPC column instead of CappedMCPC / UncappedMCPC
    raw = pd.DataFrame({
        "SCEDTimestamp": ["07/27/2026 23:55:18", "07/27/2026 23:55:18"],
        "RepeatedHourFlag": ["N", "N"],
        "ASType": ["ECRS", "NSPIN"],
        "MCPC": [0.14, 0.57],
    })
    out = get_as_prices(FakeArchive(raw), "2026-07-27", cache=tmp_path).set_index("ASType")
    assert out.loc["NSPIN", "mcpc"] == 0.57             # the single price is the one compared with ERCOT
    assert out["mcpc_uncapped"].isna().all()           # capped or not isn't stated: left empty


def test_get_as_prices_unknown_format_raises(tmp_path):
    raw = pd.DataFrame({"SCEDTimestamp": ["07/27/2026 23:55:18"], "RepeatedHourFlag": ["N"], "ASType": ["ECRS"],
                        "Price": [0.14]})
    try:
        get_as_prices(FakeArchive(raw), "2026-07-27", cache=tmp_path)
    except ValueError as e:
        assert "no MCPC column" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_get_as_demand_curves_keeps_latest_posting(tmp_path):
    rows = []
    # Hour 1 RRS posted twice: the later posting (prices 200) must win; hour 1 ECRS posted once
    for post, price in [("2026-09-21T04:55:00", 100.0), ("2026-09-21T09:00:00", 200.0)]:
        for point, mw in [(1, 0), (2, 500)]:
            rows.append(["09/21/2026", "01:00", "RRS", point, mw, price, "N", post])
    for point, mw in [(2, 345), (1, 0)]:  # deliberately out of order
        rows.append(["09/21/2026", "01:00", "ECRS", point, mw, 50.0, "N", "2026-09-21T04:55:00"])
    rows.append(["09/22/2026", "01:00", "RRS", 1, 0, 1.0, "N", "2026-09-22T04:55:00"])  # other day
    cols = ["DeliveryDate", "HourEnding", "ASType", "DemandCurvePoint", "Quantity", "Price", "RepeatedHourFlag",
            "postDatetime"]
    api = FakeArchive(pd.DataFrame(rows, columns=cols))

    curves = get_as_demand_curves(api, "2026-09-21", cache=tmp_path).set_index("ASType")
    assert len(curves) == 2                                     # one curve per (hour, product), Sept 21 only
    assert curves.loc["RRS", "curve"] == [(0.0, 200.0), (500.0, 200.0)]   # latest posting kept
    assert curves.loc["ECRS", "curve"] == [(0.0, 50.0), (345.0, 50.0)]    # points sorted by MW

def test_get_as_prices_drops_duplicate_runs(tmp_path):
    # 2026-08-27 lists some runs twice, both copies without a price: one row per run and product must remain.
    # Also: a priced copy beside an unpriced one keeps the price.
    raw = pd.DataFrame({
        "SCEDTimestamp": ["08/27/2026 10:35:17"] * 2 + ["08/27/2026 10:40:18"] * 2 + ["08/27/2026 10:45:18"],
        "RepeatedHourFlag": ["N"] * 5,
        "ASType": ["ECRS"] * 5,
        "CappedMCPC": [None, None, None, 3.5, 2.0],
        "UncappedMCPC": [None, None, None, 3.5, 2.0],
    })
    out = get_as_prices(FakeArchive(raw), "2026-08-27", cache=tmp_path)
    assert len(out) == 3
    assert not out.duplicated(["SCED Timestamp", "ASType"]).any()
    p = out.set_index("SCED Timestamp")["mcpc"]
    assert pd.isna(p.iloc[0]) and p.iloc[1] == 3.5 and p.iloc[2] == 2.0


def test_get_as_prices_cleans_an_old_cache(tmp_path):
    # A cache written before the fix still has the duplicates: reading it returns one row per run and product
    ts = pd.Timestamp("2026-08-27 10:35:17", tz="US/Central")
    pd.DataFrame({"SCED Timestamp": [ts, ts], "ASType": ["RRS", "RRS"], "mcpc": [float("nan")] * 2,
                  "mcpc_uncapped": [float("nan")] * 2}).to_pickle(tmp_path / "mcpc_2026-08-27.pkl")
    assert len(get_as_prices(FakeArchive(None), "2026-08-27", cache=tmp_path)) == 1


def test_get_as_prices_conflicting_duplicates_raise(tmp_path):
    raw = pd.DataFrame({"SCEDTimestamp": ["08/27/2026 10:35:17"] * 2, "RepeatedHourFlag": ["N"] * 2,
                        "ASType": ["NSPIN"] * 2, "CappedMCPC": [1.0, 2.0], "UncappedMCPC": [1.0, 2.0]})
    try:
        get_as_prices(FakeArchive(raw), "2026-08-27", cache=tmp_path)
    except ValueError as e:
        assert "different MCPCs" in str(e)
    else:
        raise AssertionError("expected ValueError")
