import pandas as pd

from scedrep.network import unit_nodes


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