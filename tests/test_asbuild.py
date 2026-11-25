import numpy as np
import pandas as pd
import pytest
from pytest import approx

from scedrep.asbuild import (
    as_awards,
    as_offer_blocks,
    asdc_segments,
    build_as_interval,
    hour_key,
)

TS = pd.Timestamp("2026-09-21 00:00:20", tz="US/Central")
SUB = ["REGUP", "REGDN", "RRSPFR", "RRSUFR", "RRSFFR", "NSPIN", "ECRS"]


def offer_row(name, otype, q, prices, ts=TS, suffix=None):
    """One raw AS offer row: q = {block: MW}, prices = {(block, product): $/MW-h}; the rest missing."""
    r = {"SCED Timestamp": ts, "Resource Name": name, "Offer Type": otype}
    for b in range(1, 7):
        r[f"QUANTITY_MW{b}"] = q.get(b, 0.0)            # unused blocks carry 0 MW, as in ERCOT's files
        for s in SUB:
            col = f"PRICE{b}_{(suffix or {}).get(s, s)}"
            r[col] = prices.get((b, s), np.nan)         # not offered = missing
    return r


def test_offer_blocks_shared_mw_and_pooled_rrs():
    offers = pd.DataFrame([
        # block 1: 20 MW shared by RegUp and two RRS sub-types; block 2: priced but 0 MW (dropped);
        # block 3: 30 MW of ECRS only
        offer_row("G", "ONRES", {1: 20, 3: 30},
                  {(1, "REGUP"): 5, (1, "RRSPFR"): 3, (1, "RRSUFR"): 2, (2, "NSPIN"): 9, (3, "ECRS"): 10}),
        offer_row("G", "OFFNS", {1: 40}, {(1, "NSPIN"): 1}),    # OFFNS rows are not used
        offer_row("G", "REGDN", {1: 50}, {(1, "REGDN"): 1}),
    ])
    x = as_offer_blocks(offers).set_index(["Offer Type", "block", "product"])

    assert set(x.index) == {("ONRES", 1, "REGUP"), ("ONRES", 1, "RRS"), ("ONRES", 3, "ECRS"),
                            ("REGDN", 1, "REGDN")}
    assert x.loc[("ONRES", 1, "RRS"), "price"] == 2           # cheapest RRS sub-type
    assert x.loc[("ONRES", 1, "REGUP"), "mw"] == 20           # same block, same shared MW
    assert x.loc[("ONRES", 1, "RRS"), "mw"] == 20


def test_offer_blocks_old_suffixes_read_the_same():
    old = {"REGUP": "URS", "REGDN": "DRS", "RRSPFR": "RRSPF", "RRSUFR": "RRSUF", "RRSFFR": "RRSFF", "NSPIN": "NS"}
    prices = {(1, "REGUP"): 5, (1, "NSPIN"): 4}
    new = as_offer_blocks(pd.DataFrame([offer_row("G", "ONRES", {1: 20}, prices)]))
    olds = as_offer_blocks(pd.DataFrame([offer_row("G", "ONRES", {1: 20}, prices, suffix=old)]))
    pd.testing.assert_frame_equal(new, olds)


def test_offer_blocks_guards():
    row = offer_row("G", "ONRES", {1: 20}, {(1, "REGUP"): 5})
    no_type = pd.DataFrame([row]).drop(columns="Offer Type")
    with pytest.raises(ValueError, match="Offer Type"):
        as_offer_blocks(no_type)
    zero_coded = pd.DataFrame([row]).fillna(0)                # "not offered" written as 0
    with pytest.raises(ValueError, match="coded as 0"):
        as_offer_blocks(zero_coded)


def test_awards_processed_and_raw_names():
    processed = pd.DataFrame({"SCED Timestamp": [TS], "Resource Name": ["G"], "AS Awards RegUp": [1.0],
                              "AS Awards RegDown": [2.0], "AS Awards ECRS": [3.0], "AS Awards NonSpin": [4.0],
                              "AS Awards RRSPFR": [5.0], "AS Awards RRSUFR": [6.0], "AS Awards RRSFFR": [np.nan]})
    raw = processed.rename(columns={"AS Awards RegUp": "AS Awards REGUP", "AS Awards RegDown": "AS Awards REGDN",
                                    "AS Awards NonSpin": "AS Awards NSPIN"})
    for t in (processed, raw):
        a = as_awards(t).iloc[0]
        assert (a["REGUP"], a["REGDN"], a["ECRS"], a["NSPIN"], a["RRS"]) == (1, 2, 3, 4, 11)  # RRS = 5 + 6
    with pytest.raises(ValueError, match="ECRS"):
        as_awards(processed.drop(columns="AS Awards ECRS"))


def test_asdc_segments():
    # flat cap to 301 MW, a 1-MW cliff, then a sloped tail to the AS plan at 334 MW
    segs = asdc_segments([(0, 9052), (301, 9052), (302, 3355.43), (334, 250.26)])
    assert segs[0] == (9052, 0, 301)
    assert segs[1][0] == 9052 and segs[1][1] == approx(3355.43 - 9052) and segs[1][2] == 1
    assert segs[2][1] == approx((250.26 - 3355.43) / 32)
    assert asdc_segments([(0, 50), (10, 50), (10, 40), (20, 30)]) == [(50, 0, 10), (40, -1, 10)]  # step skipped
    with pytest.raises(ValueError, match="concave"):
        asdc_segments([(0, 10), (5, 20)])


def test_hour_key():
    assert hour_key(TS) == ("01:00", "N")
    assert hour_key(pd.Timestamp("2026-09-21 23:55:10", tz="US/Central")) == ("24:00", "N")
    # fall-back day 2026-11-01: 01:30 occurs in daylight time (-05:00), then again in standard time (-06:00)
    assert hour_key(pd.Timestamp("2026-11-01 06:30", tz="UTC")) == ("02:00", "N")   # 01:30 CDT
    assert hour_key(pd.Timestamp("2026-11-01 07:30", tz="UTC")) == ("02:00", "Y")   # 01:30 CST


def test_build_as_interval_fixed_supply_and_blocks():
    offers = pd.DataFrame([
        offer_row("A", "ONRES", {1: 20}, {(1, "REGUP"): 5}),
        offer_row("OFF1", "ONRES", {1: 40}, {(1, "NSPIN"): 2}),   # offline: not dispatched by the model
    ])
    blocks = as_offer_blocks(offers)
    gen = pd.DataFrame({"SCED Timestamp": [TS] * 2, "Resource Name": ["A", "OFF1"],
                        "AS Awards RegUp": [10.0, 0.0], "AS Awards RegDown": [0.0, 0.0], "AS Awards ECRS": [0.0, 0.0],
                        "AS Awards NonSpin": [0.0, 25.0], "AS Awards RRSPFR": [0.0, 0.0],
                        "AS Awards RRSUFR": [0.0, 0.0], "AS Awards RRSFFR": [0.0, 0.0]})
    load = gen.iloc[:1].assign(**{"Resource Name": "L1", "AS Awards RegUp": 0.0, "AS Awards RRSUFR": 9.0})
    awards = as_awards(gen, load)
    asdc = pd.DataFrame([{"HourEnding": "01:00", "RepeatedHourFlag": "N", "ASType": k,
                          "curve": [(0, 5000), (100, 5000)]} for k in ["REGUP", "REGDN", "RRS", "ECRS", "NSPIN"]])

    out = build_as_interval(blocks, awards, asdc, TS, units={"A": {}})

    assert set(out["blocks"]["Resource Name"]) == {"A"}           # only dispatched units' blocks
    assert out["fixed"] == {"REGUP": 0, "REGDN": 0, "RRS": 9, "ECRS": 0, "NSPIN": 25}  # load + offline unit
    assert out["demand"]["REGUP"] == [(5000, 0, 100)]