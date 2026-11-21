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

def congestion_matrices(lmp, lam, sp):
    # build one day's regression matrices.
    # Resource nodes only; hubs and load zones are averages over many buses
    lmp = lmp[lmp["Location Type"] == "Resource Node"]
    # Congestion component of each node in each run: LMP_n - lambda
    lmp = lmp.merge(lam[["SCED Timestamp", "lam"]], on="SCED Timestamp")
    lmp = lmp.assign(cong=lmp["LMP"] - lmp["lam"])
    Y = lmp.pivot_table(index="SCED Timestamp", columns="Location", values="cong").fillna(0.0)
    # Binding constraints, keyed as "monitored element | contingency"
    b = sp[sp["Shadow Price"] > 0]
    b = b.assign(key=b["Constraint Name"] + " | " + b["Contingency Name"])
    M = b.pivot_table(index="SCED Timestamp", columns="key", values="Shadow Price", aggfunc="sum")
    # Same runs as Y; a constraint that didn't bind in a run has mu = 0
    M = M.reindex(Y.index).fillna(0.0)
    return Y, M

def build_network(bp, node, SF, sp_run):
    """Network constraints for one SCED run, in the form solve_interval takes.

    bp:     base point of each dispatchable unit (Series: unit -> MW).
    node:   resource node of each unit (Series: unit -> node), from unit_nodes.
    SF:     estimated shift factors (constraints x nodes), from fit_shift_factors.
    sp_run: this run's rows of the SCED shadow-price table.
    Returns a list of {"sf": {unit: SF}, "rhs": float, "penalty": float}.
    """
    network = []

    # Only constraints ERCOT found binding in this run
    b = sp_run[sp_run["Shadow Price"] > 0]
    for _, r in b.iterrows():
        key = r["Constraint Name"] + " | " + r["Contingency Name"]

        # No estimate if it never bound in the training window: leave it out
        if key not in SF.index:
            continue

        # A unit's shift factor is its node's; no node or no estimate means 0
        sf = node.map(SF.loc[key]).reindex(bp.index).fillna(0.0)

        # Anchor at ERCOT's dispatch: rhs = (Limit - Flow) + sum_u SF_u * BP_u
        rhs = (r["Limit"] - r["Value"]) + float((sf * bp).sum())

        # Keep nonzero terms only; penalty = ERCOT's shadow-price cap for this constraint
        network.append({"sf": sf[sf != 0].to_dict(), "rhs": rhs, "penalty": float(r["Max Shadow Price"])})

    return network