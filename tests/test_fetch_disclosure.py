import pandas as pd

from scedrep import fetch


class FakeErcot:
    """Stands in for gridstatus.Ercot: returns canned raw tables and records how it was called."""

    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def get_60_day_sced_disclosure(self, date, process=False):
        self.calls.append((date, process))
        return self.raw


def tag(name):
    """Stub for a gridstatus processing function: marks the table so the test can see it was applied."""
    return lambda df: df.assign(processed_by=name)


def test_one_raw_download_resource_tables_processed_offers_raw(monkeypatch):
    monkeypatch.setattr(fetch, "process_sced_gen", tag("gen"))
    monkeypatch.setattr(fetch, "process_sced_esr", tag("esr"))
    monkeypatch.setattr(fetch, "process_sced_load", tag("load"))
    offers = pd.DataFrame({"Resource Name": ["G"], "QUANTITY_MW1": [20.0], "PRICE1_REGUP": [5.0]})
    raw = {k: pd.DataFrame({"Resource Name": ["G"]})
           for k in ["sced_gen_resource", "sced_esr", "sced_load_resource"]}
    raw["sced_resource_as_offers"] = offers
    api = FakeErcot(raw)

    out = fetch.get_sced_disclosure("2026-09-21", ercot=api)

    assert api.calls == [("2026-09-21", False)]                     # one download, raw
    assert out["sced_gen_resource"]["processed_by"].item() == "gen"
    assert out["sced_esr"]["processed_by"].item() == "esr"
    assert out["sced_load_resource"]["processed_by"].item() == "load"
    pd.testing.assert_frame_equal(out["sced_resource_as_offers"], offers)  # untouched: blocks kept


def test_missing_tables_are_none(monkeypatch):
    monkeypatch.setattr(fetch, "process_sced_gen", tag("gen"))
    monkeypatch.setattr(fetch, "process_sced_load", tag("load"))
    raw = {"sced_gen_resource": pd.DataFrame({"a": [1]}), "sced_load_resource": pd.DataFrame({"a": [1]})}

    out = fetch.get_sced_disclosure("2025-11-01", ercot=FakeErcot(raw))   # before batteries / RTC+B tables

    assert out["sced_esr"] is None
    assert out["sced_resource_as_offers"] is None