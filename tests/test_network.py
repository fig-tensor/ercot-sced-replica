import numpy as np
import pandas as pd

from scedrep.network import fit_shift_factors, unit_nodes


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