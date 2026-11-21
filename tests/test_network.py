import numpy as np
import pandas as pd

from scedrep.network import congestion_matrices, fit_shift_factors, unit_nodes


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
    # Two SCED runs with lambda 30 and 40
    ts = pd.to_datetime(["2026-09-08 00:00:15", "2026-09-08 00:05:15"]).tz_localize("US/Central")
    lam = pd.DataFrame({"SCED Timestamp": ts, "lam": [30.0, 40.0]})

    # Two resource nodes plus a hub, which should be dropped
    lmp = pd.DataFrame({
        "SCED Timestamp": [ts[0], ts[0], ts[1], ts[1], ts[0]],
        "Location": ["N1", "N2", "N1", "N2", "HB_HOUSTON"],
        "Location Type": ["Resource Node"] * 4 + ["Trading Hub"],
        "LMP": [30.0, 25.0, 45.0, 40.0, 29.0],
    })

    # One constraint binds in run 2; a zero shadow price means not binding
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