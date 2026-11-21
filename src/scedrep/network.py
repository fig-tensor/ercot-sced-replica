import numpy as np
import pandas as pd


def unit_nodes(names, rn, ccp):

    names = pd.Series(names)
    # Lookup tables: "<substation>_<unit>" -> node, and combined-cycle train -> node
    direct = dict(zip(rn["Unit Substation"] + "_" + rn["Unit Name"], rn["Resource Node"], strict=True))
    cc = dict(zip(ccp["CCP Name"], ccp["Logical Resource Node Name"], strict=True))
    # resource name is "<substation>_<unit>"
    node = names.map(direct)
    # combined-cycle configuration; strip "_<n>" to get the train (AMOCOOIL_CC1_3 -> AMOCOOIL_CC1)
    node = node.fillna(names.str.replace(r"_\d+$", "", regex=True).map(cc))
    # jointly owned unit share; strip "_J<n>" to get the unit (HICKMAN_G1_J02 -> HICKMAN_G1)
    node = node.fillna(names.str.replace(r"_J\d+$", "", regex=True).map(direct))
    # index by resource name so node["AMOCOOIL_CC1_3"] returns "AMOCOOIL_CC1"
    return pd.Series(node.values, index=names.values)

def fit_shift_factors(Ys, Ms):
    # estimate shift factors from several days of congestion data.
    # stack the days; keep only nodes present on every day
    Y = pd.concat(Ys, join="inner")
    # stack shadow prices; a constraint missing on a day didn't bind, so mu = 0.
    # reindex to Y's runs so row i of M and row i of Y are the same SCED run.
    M = pd.concat(Ms).reindex(Y.index).fillna(0.0)
    # least squares for every node at once: minimize ||Y - M @ B||^2 over B
    B = np.linalg.lstsq(M.values, Y.values, rcond=None)[0]
    # the model is Y = -M @ SF, so SF = -B
    return pd.DataFrame(-B, index=M.columns, columns=Y.columns)