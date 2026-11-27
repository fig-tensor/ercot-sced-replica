import os

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

def congestion_matrices(lmp, lam, sp, types=("Resource Node",)):
    # build one day's regression matrices.
    # Resource nodes only; hubs and load zones are averages over many buses
    # Which settlement points to keep, by gridstatus "Location Type": resource nodes by default;
    # ("Trading Hub", "Load Zone") gives the hubs and load zones (each an average over many buses)
    lmp = lmp[lmp["Location Type"].isin(types)]
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

def build_network(bp, node, SF, sp_run, sf_min=0.0):
    """Network constraints for one SCED run, in the form solve_interval takes.

    bp:     base point of each dispatchable unit (Series: unit -> MW).
    node:   resource node of each unit (Series: unit -> node), from unit_nodes.
    SF:     estimated shift factors (constraints x nodes), from fit_shift_factors.
    sp_run: this run's rows of the SCED shadow-price table.
    types: which settlement point types to keep (gridstatus "Location Type" values).
    sf_min: shift factors with |SF| below this are set to 0 (regression noise).
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

        # Drop tiny shift factors: mostly noise, and they make the QP badly scaled
        sf = sf.where(sf.abs() >= sf_min, 0.0)

        # Anchor at ERCOT's dispatch: rhs = (Limit - Flow) + sum_u SF_u * BP_u
        rhs = (r["Limit"] - r["Value"]) + float((sf * bp).sum())

        # Keep nonzero terms only; penalty = ERCOT's shadow-price cap for this constraint
        network.append({"sf": sf[sf != 0].to_dict(), "rhs": rhs, "penalty": float(r["Max Shadow Price"])})

    return network

def unit_capacity(gen, node, types=("WIND", "PVGR")):
    """Type, node and capacity of each unit of the given types, from one day of 60-day SCED data.

    gen:  the day's sced_gen_resource table.
    node: resource node of each unit (Series: unit -> node), from unit_nodes.
    Capacity is the unit's highest HSL that day (MW). Units with no known node are dropped.
    """
    g = gen[gen["Resource Type"].isin(types)]
    units = g.groupby("Resource Name").agg(type=("Resource Type", "first"), capacity=("HSL", "max"))
    units["node"] = node.reindex(units.index).to_numpy()
    return units.dropna(subset=["node"])


def type_sensitivity(SF, units):
    """Flow change on each constraint per MW of each unit type, with output spread by capacity.

    s[c, k] = sum over units u of type k of SF[c, n(u)] * cap_u / (sum of cap_u over type k).
    Units whose node has no shift-factor estimate count as 0.
    Returns constraints x types.
    """
    out = {}
    for k, g in units.groupby("type"):
        w = g["capacity"] / g["capacity"].sum()             # each unit's share of the type's capacity
        S = SF.reindex(columns=g["node"]).fillna(0.0)       # SF[c, n(u)] for each unit, in unit order
        out[k] = S.to_numpy() @ w.to_numpy()                # capacity-weighted average shift factor
    return pd.DataFrame(out, index=SF.index)

def export_network(SF, node, out_dir, label):
    """Write the fitted network for others to use: shift factors and the resource-to-settlement-point map.

    SF:    shift factors from fit_shift_factors (constraints x settlement points).
    node:  resource -> settlement point (Series), from unit_nodes; resources without a node are left out.
    label: tag for the file names, e.g. the run date the fit is for.
    Writes shift_factors_{label}.csv.gz (one row per constraint, keyed "constraint | contingency"; one column per
    settlement point) and resource_nodes_{label}.csv (resource, settlement_point). Returns the two paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    sf_path = os.path.join(out_dir, f"shift_factors_{label}.csv.gz")
    rn_path = os.path.join(out_dir, f"resource_nodes_{label}.csv")
    SF.rename_axis("constraint").to_csv(sf_path, float_format="%.6g")
    nodes = node.dropna().rename_axis("resource").rename("settlement_point").sort_index()
    nodes.to_csv(rn_path)
    return sf_path, rn_path
