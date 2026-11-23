import numpy as np
import pandas as pd
import pytest

from scedrep.network import (
    build_network,
    congestion_matrices,
    fit_shift_factors,
    type_sensitivity,
    unit_capacity,
    unit_nodes,
)


def test_unit_nodes():
    # toy mapping tables: two plain units and one combined-cycle train
    rn = pd.DataFrame({"Resource Node": ["A_RN", "H_RN"], "Unit Substation": ["A", "HICK"], "Unit Name": ["G1", "G1"]})
    ccp = pd.DataFrame({"CCP Name": ["B_CC1"], "Logical Resource Node Name": ["B_CC1"]})
    # one name per rule, plus one that matches nothing
    node = unit_nodes(["A_G1", "B_CC1_3", "HICK_G1_J02", "X_SOLAR1"], rn, ccp)

    assert node["A_G1"] == "A_RN"         # direct match
    assert node["B_CC1_3"] == "B_CC1"     # combined-cycle configuration
    assert node["HICK_G1_J02"] == "H_RN"  # jointly owned unit share
    assert pd.isna(node["X_SOLAR1"])      # unmatched stays NaN

def test_fit_shift_factors():
    # known shift factors: 2 constraints x 3 nodes
    SF = pd.DataFrame([[0.5, -0.2, 0.0], [0.1, 0.3, -0.4]], index=["c1", "c2"], columns=["n1", "n2", "n3"])
    # two days of shadow prices; c2 never binds on day 1, so it has no column there
    M1 = pd.DataFrame({"c1": [10.0, 0.0, 5.0]}, index=[0, 1, 2])
    M2 = pd.DataFrame({"c1": [0.0, 8.0], "c2": [4.0, 6.0]}, index=[3, 4])
    # congestion components the model implies: Y = -M @ SF
    Y1 = -M1.reindex(columns=SF.index, fill_value=0.0) @ SF
    Y2 = -M2 @ SF
    # the fit should recover SF exactly (no noise in the toy data)
    est = fit_shift_factors([Y1, Y2], [M1, M2])
   
    assert np.allclose(est.loc[SF.index, SF.columns], SF)

def test_congestion_matrices():
    # two SCED runs with lambda 30 and 40
    ts = pd.to_datetime(["2026-09-08 00:00:15", "2026-09-08 00:05:15"]).tz_localize("US/Central")
    lam = pd.DataFrame({"SCED Timestamp": ts, "lam": [30.0, 40.0]})
    # two resource nodes plus a hub, which should be dropped
    lmp = pd.DataFrame({
        "SCED Timestamp": [ts[0], ts[0], ts[1], ts[1], ts[0]],
        "Location": ["N1", "N2", "N1", "N2", "HB_HOUSTON"],
        "Location Type": ["Resource Node"] * 4 + ["Trading Hub"],
        "LMP": [30.0, 25.0, 45.0, 40.0, 29.0],
    })
    # one constraint binds in run 2; a zero shadow price means not binding
    sp = pd.DataFrame({
        "SCED Timestamp": [ts[1], ts[1]],
        "Constraint Name": ["L1", "L2"],
        "Contingency Name": ["K1", "K2"],
        "Shadow Price": [12.0, 0.0],
    })
    Y, M = congestion_matrices(lmp, lam, sp)

    assert list(Y.columns) == ["N1", "N2"]         # hub dropped
    assert Y.loc[ts[0]].tolist() == [0.0, -5.0]    # LMP - lambda, run 1
    assert Y.loc[ts[1]].tolist() == [5.0, 0.0]     # LMP - lambda, run 2
    assert list(M.columns) == ["L1 | K1"]          # non-binding L2 dropped
    assert M["L1 | K1"].tolist() == [0.0, 12.0]    # mu = 0 when not binding

def test_build_network():
    # base points: ERCOT dispatched A at 60 and B at 40; C has no known node
    bp = pd.Series({"A": 60.0, "B": 40.0, "C": 5.0})
    node = pd.Series({"A": "N1", "B": "N2"})
    # estimated shift factors for one constraint
    SF = pd.DataFrame({"N1": [1.0], "N2": [0.0]}, index=["L1 | K1"])
    # L1 binds at its limit; L9 binds but has no estimate; L2 isn't binding
    sp_run = pd.DataFrame({
        "Constraint Name": ["L1", "L9", "L2"],
        "Contingency Name": ["K1", "K9", "K2"],
        "Shadow Price": [30.0, 12.0, 0.0],
        "Limit": [60.0, 100.0, 50.0],
        "Value": [60.0, 100.0, 20.0],
        "Max Shadow Price": [1000.0, 1000.0, 1000.0],
    })
    net = build_network(bp, node, SF, sp_run)
    
    # only L1 survives; rhs = (60 - 60) + 1 * 60 = 60, so A can't exceed ERCOT's 60 MW
    assert len(net) == 1
    assert net[0]["sf"] == {"A": 1.0}
    assert net[0]["rhs"] == 60.0
    assert net[0]["penalty"] == 1000.0

def test_build_network_drops_tiny_shift_factors():
    # Node N2's shift factor (0.004) is below the cutoff, so B drops out of the constraint
    bp = pd.Series({"A": 60.0, "B": 40.0})
    node = pd.Series({"A": "N1", "B": "N2"})
    SF = pd.DataFrame({"N1": [1.0], "N2": [0.004]}, index=["L1 | K1"])
    sp_run = pd.DataFrame({
        "Constraint Name": ["L1"],
        "Contingency Name": ["K1"],
        "Shadow Price": [30.0],
        "Limit": [60.0],
        "Value": [60.0],
        "Max Shadow Price": [1000.0],
    })

    net = build_network(bp, node, SF, sp_run, sf_min=0.01)

    # B is gone, and rhs uses the same (cut) shift factors, so ERCOT's dispatch stays feasible
    assert net[0]["sf"] == {"A": 1.0}
    assert net[0]["rhs"] == 60.0

def test_unit_capacity():
    # Two SCED runs; W1's HSL varies (capacity = the max), G1 is gas (excluded), W2 has no node (dropped)
    gen = pd.DataFrame({
        "Resource Name": ["W1", "W1", "S1", "G1", "W2"],
        "Resource Type": ["WIND", "WIND", "PVGR", "CCGT90", "WIND"],
        "HSL": [80.0, 100.0, 50.0, 500.0, 30.0],
    })
    node = pd.Series({"W1": "N1", "S1": "N2", "G1": "N3"})

    units = unit_capacity(gen, node)

    assert sorted(units.index) == ["S1", "W1"]
    assert units.loc["W1", "capacity"] == 100.0
    assert units.loc["W1", "node"] == "N1"
    assert units.loc["S1", "type"] == "PVGR"


def test_type_sensitivity():
    # One constraint: SF 1.0 at N1, 0.2 at N2; N9 has no estimate
    SF = pd.DataFrame({"N1": [1.0], "N2": [0.2]}, index=["L1 | K1"])
    units = pd.DataFrame({
        "type": ["WIND", "WIND", "WIND", "PVGR"],
        "capacity": [100.0, 300.0, 100.0, 50.0],
        "node": ["N1", "N2", "N9", "N1"],
    }, index=["W1", "W2", "W3", "S1"])

    s = type_sensitivity(SF, units)

    # Wind: (1.0 * 100 + 0.2 * 300 + 0 * 100) / 500 = 0.32; solar: all at N1, so 1.0
    assert s.loc["L1 | K1", "WIND"] == pytest.approx(0.32)
    assert s.loc["L1 | K1", "PVGR"] == pytest.approx(1.0)