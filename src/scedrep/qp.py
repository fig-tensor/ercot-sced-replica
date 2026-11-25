"""The SCED model as matrices, solved with PIQP (interior point).

Same model as dispatch.solve_interval's Pyomo version, written in PIQP's standard form

    min 1/2 z'Pz + c'z   s.t.   A z = b,   G z <= h,   z_lb <= z <= z_ub

Variables z, in this order: energy segment fills x[u,j]; outputs p[u]; shortage s; network violations v[c];
AS block awards xa[u,o,b,k]; demand-segment fills y[k,i]. Prices are read from the duals: lambda and MCPC are
-y of the energy and AS balance rows, mu is the multiplier of each network row.
"""

import numpy as np
import piqp
import scipy.sparse as sp

from scedrep.dispatch import (
    DT,
    PFR_PENALTY,
    RAMP,
    UP,
    SolveError,
    as_terms,
    energy_segments,
)

__all__ = ["SolveError", "solve_interval_piqp"]  # SolveError lives in dispatch; re-exported for old imports


class _Rows:
    """Collects sparse rows (column index -> coefficient) and their right-hand sides."""

    def __init__(self):
        self.r, self.c, self.v, self.rhs = [], [], [], []

    def add(self, coefs, rhs):
        """Append one row sum_j coefs[j] * z[j] (= or <=) rhs; returns its row index."""
        i = len(self.rhs)
        for j, a in coefs.items():
            self.r.append(i)
            self.c.append(j)
            self.v.append(a)
        self.rhs.append(rhs)
        return i

    def matrix(self, n):
        return sp.csc_matrix((self.v, (self.r, self.c)), shape=(len(self.rhs), n)), np.array(self.rhs, dtype=float)


def solve_interval_piqp(units, load, penalty, network=(), verbose=False, asin=None, durations=None, settings=None):
    """Solve one SCED run with PIQP. Same arguments and output as dispatch.solve_interval."""
    segs = energy_segments(units)
    names = list(units)

    # ---- variable index: one column per variable, in a fixed order ----
    col = {}
    for key in segs:
        col["x", *key] = len(col)
    for u in names:
        col["p", u] = len(col)
    col["s"] = len(col)
    for c in range(len(network)):
        col["v", c] = len(col)
    t = as_terms(units, asin) if asin is not None else None
    if t is not None:
        for key in t["price"]:
            col["xa", *key] = len(col)
        for key in t["seg"]:
            col["y", *key] = len(col)
        if t["pfr"]:
            col["pfr_short"] = len(col)                       # MW short of the PFR floor
    n = len(col)

    # ---- objective 1/2 z'Pz + c'z (P diagonal) and bounds ----
    Pd, cost = np.zeros(n), np.zeros(n)
    lb, ub = np.zeros(n), np.full(n, np.inf)
    for (u, k), (w, price, slope) in segs.items():
        j = col["x", u, k]
        Pd[j], cost[j], ub[j] = slope, price, w               # cost price*x + slope*x^2/2, 0 <= x <= width
    for u in names:
        j = col["p", u]
        lb[j], ub[j] = units[u]["ldl"], units[u]["hdl"]       # LDL <= p <= HDL
    cost[col["s"]] = penalty                                  # unserved energy at the penalty price
    for c in range(len(network)):
        cost[col["v", c]] = network[c]["penalty"]             # line violation at its MaxSP
    if t is not None:
        for key, pr in t["price"].items():
            cost[col["xa", *key]] = pr                        # AS offer price per MW awarded
        for (k, i), (P_, s_, w_) in t["seg"].items():
            j = col["y", k, i]
            Pd[j], cost[j], ub[j] = -s_, -P_, w_              # minus the concave value P*y + s*y^2/2
        if t["pfr"]:
            cost[col["pfr_short"]] = PFR_PENALTY              # shortfall priced at PFR_PENALTY

    # ---- equalities A z = b ----
    eq = _Rows()
    by_unit = {u: [] for u in names}
    for u, k in segs:
        by_unit[u].append(k)
    for u in names:                                           # link: p - sum_j x[u,j] = curve start
        coefs = {col["p", u]: 1.0, **{col["x", u, k]: -1.0 for k in by_unit[u]}}
        eq.add(coefs, units[u]["curve"][0][0])
    bal = eq.add({**{col["p", u]: 1.0 for u in names}, col["s"]: 1.0}, load)   # energy balance -> lambda
    as_rows = {}
    if t is not None:
        for k in t["K"]:                                      # AS balance: sum xa - sum y = -F_k -> MCPC_k
            coefs = {col["xa", *key]: 1.0 for u in t["as_units"] for key in t["by_award"].get((u, k), [])}
            coefs.update({col["y", k, i]: -1.0 for (kk, i) in t["seg"] if kk == k})
            if coefs:                                         # nothing to clear for this product: no row
                as_rows[k] = eq.add(coefs, -t["fixed"].get(k, 0.0))

    # ---- inequalities G z <= h ----
    ineq = _Rows()
    net_rows = []
    for c, row in enumerate(network):                         # sum_u SF*p - v <= rhs -> mu_c
        coefs = {col["p", u]: sf for u, sf in row["sf"].items() if u in units}
        coefs[col["v", c]] = -1.0
        net_rows.append(ineq.add(coefs, row["rhs"]))
    if t is not None:
        def award(u, k, a=1.0):
            """Coefficients of a * (award of product k to unit u)."""
            return {col["xa", *key]: a for key in t["by_award"].get((u, k), [])}

        for blk, keys in t["by_block"].items():               # block capacity: products share Q
            ineq.add({col["xa", *key]: 1.0 for key in keys}, t["Q"][blk])
        for u in t["as_units"]:
            p, un = col["p", u], units[u]
            up = {}
            for k in UP:
                up.update(award(u, k))
            ineq.add({p: 1.0, **up}, t["hsl"][u])                               # p + up-awards <= HSL
            ineq.add({p: -1.0, **award(u, "REGDN")}, -t["lsl"][u])             # -p + RegDn <= -LSL
            ineq.add({p: 1.0, **award(u, "REGUP", RAMP)}, un["hdl"])           # p + 5/7 RegUp <= HDL
            ineq.add({p: -1.0, **award(u, "REGDN", RAMP)}, -un["ldl"])         # -p + 5/7 RegDn <= -LDL
            if durations and not np.isnan(un["soc"]):                           # battery duration rows
                coefs = {p: DT}
                for k in UP:
                    for j, a in award(u, k, durations.get(k, 0.0)).items():
                        coefs[j] = coefs.get(j, 0.0) + a
                ineq.add(coefs, un["soc"] - un["soc_min"])
                ineq.add({p: -DT, **award(u, "REGDN", durations.get("REGDN", 0.0))}, un["soc_max"] - un["soc"])

        for (u, k), cap in t["award_cap"].items():           # award caps: ramp minutes, PFR per resource
            ineq.add(award(u, k), cap)
        if t["pfr"]:                                          # PFR floor: -sum PFR - short <= F_PFR - min
            coefs = {}
            for u in t["as_units"]:
                coefs.update(award(u, "RRSPFR", -1.0))
            coefs[col["pfr_short"]] = -1.0
            ineq.add(coefs, t["fixed"].get("RRSPFR", 0.0) - t["pfr"]["min"])

    # ---- solve ----
    A, b = eq.matrix(n)
    G, h = ineq.matrix(n)
    solver = piqp.SparseSolver()
    solver.settings.verbose = verbose
    for key, value in (settings or {}).items():
        setattr(solver.settings, key, value)
    solver.setup(sp.diags(Pd, format="csc"), cost, A, b, G, np.full(len(h), -np.inf), h, lb, ub)
    status = solver.solve()
    if status != piqp.PIQP_SOLVED:
        raise SolveError(f"PIQP status {status}")
    r = solver.result
    z = r.x

    out = {
        "dispatch": {u: float(z[col["p", u]]) for u in names},
        "lambda": float(-r.y[bal]),
        "shortage": float(z[col["s"]]),
        "mu": [float(r.z_u[i]) for i in net_rows],
        "violation": [float(z[col["v", c]]) for c in range(len(network))],
    }
    if t is not None:
        out["mcpc"] = {k: float(-r.y[i]) for k, i in as_rows.items()}
        ks = t["K"] + ["RRSPFR"] if t["pfr"] else t["K"]      # PFR awards reported alongside the products
        out["awards"] = {u: {k: float(sum(z[col["xa", *key]] for key in t["by_award"].get((u, k), [])))
                             for k in ks} for u in t["as_units"]}
        out["cleared"] = {k: float(sum(z[col["y", k, i]] for (kk, i) in t["seg"] if kk == k)) for k in t["K"]}
        if t["pfr"]:
            out["pfr_short"] = float(z[col["pfr_short"]])     # MW short of the PFR floor
    return out