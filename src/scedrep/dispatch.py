import itertools
import math
from collections import defaultdict

import pyomo.environ as pyo
from pyomo.contrib.solver.solvers.highs import Highs

DT = 5 / 60  # SCED interval length in hours
UP = ("REGUP", "RRS", "ECRS", "NSPIN")  # up-products: share headroom above p
RAMP = 5 / 7  # regulation ramp factor in the ramp rows (ERCOT training material)


def _add_as(m, units, asin, durations):
    """Add ancillary-service (AS) variables and rows to model m; return the AS part of the objective.

    asin: output of asbuild.build_as_interval -> {"blocks", "fixed", "demand"}.
    Variables: x[u,o,b,k] >= 0, MW of offer block (u, o, b) awarded for product k; y[k,i] in [0, w],
    MW filled on demand segment i of product k. Awards a[u,k] = sum of x are expressions.
    Rows: AS balance per product (dual = MCPC), block capacity, capacity and ramp rows for units that
    offer AS, and battery duration rows if durations = {product: hours} is given.
    """
    demand, fixed = asin["demand"], asin.get("fixed", {})
    K = list(demand)  # products with a demand curve this run

    # Offered (unit, offer type, block, product) -> price, and (unit, offer type, block) -> shared MW Q
    blk = asin["blocks"]
    blk = blk[blk["Resource Name"].isin(units.keys()) & blk["product"].isin(K)]
    cols = [blk["Resource Name"], blk["Offer Type"], blk["block"], blk["product"]]
    price = {(u, o, int(b), k): float(pr) for u, o, b, k, pr in zip(*cols, blk["price"], strict=True)}
    Q = {(u, o, int(b)): float(q) for u, o, b, q in zip(*cols[:3], blk["mw"], strict=True)}

    # Index the x keys by award (u, k) and by block (u, o, b), so each row sums only its own terms
    by_award, by_block = defaultdict(list), defaultdict(list)
    for key in price:
        by_award[key[0], key[3]].append(key)
        by_block[key[:3]].append(key)
    as_units = sorted({key[0] for key in price})
    for u in as_units:
        if math.isnan(units[u]["hsl"]) or math.isnan(units[u]["lsl"]):
            raise ValueError(f"unit {u} offers AS but has no HSL/LSL")

    m.X = pyo.Set(initialize=list(price), dimen=4)
    m.x_as = pyo.Var(m.X, within=pyo.NonNegativeReals)

    def a(u, k):
        """Award of product k to unit u: sum of x over its blocks offering k (0 if none)."""
        return sum(m.x_as[key] for key in by_award.get((u, k), []))

    # Block capacity: the products a block prices share its MW
    m.BLK = pyo.Set(initialize=list(by_block), dimen=3)
    m.block_cap = pyo.Constraint(m.BLK, rule=lambda m, u, o, b: sum(m.x_as[key] for key in by_block[u, o, b]) <= Q[u, o, b])

    # Demand segments (P, s, w): fill y in [0, w], worth P*y + s*y^2/2 (s <= 0: concave)
    seg = {(k, i): sw for k in K for i, sw in enumerate(demand[k])}
    m.Y = pyo.Set(initialize=list(seg), dimen=2)
    m.y = pyo.Var(m.Y, bounds=lambda m, k, i: (0, seg[k, i][2]))

    # AS balance: sum_u a[u,k] + F_k = q_k, written as supply - demand = -F so its dual is MCPC >= 0
    def balance(m, k):
        expr = sum(a(u, k) for u in as_units) - sum(m.y[k, i] for i in range(len(demand[k])))
        if isinstance(expr, (int, float)):  # no supply and no demand segments: nothing to clear
            return pyo.Constraint.Skip
        return expr == -fixed.get(k, 0.0)
    m.K = pyo.Set(initialize=K)
    m.as_balance = pyo.Constraint(m.K, rule=balance)

    # Capacity and ramp rows, only for units that offer AS
    m.AU = pyo.Set(initialize=as_units)
    up = lambda u: sum(a(u, k) for k in UP)
    m.cap_hi = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] + up(u) <= units[u]["hsl"])
    m.cap_lo = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] - a(u, "REGDN") >= units[u]["lsl"])
    m.ramp_hi = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] + RAMP * a(u, "REGUP") <= units[u]["hdl"])
    m.ramp_lo = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] - RAMP * a(u, "REGDN") >= units[u]["ldl"])

    # Battery duration: energy out this interval + energy to sustain each up-award for d_k hours
    # <= stored energy above the minimum; mirror on the charging side for RegDn
    if durations:
        bats = [u for u in as_units if not math.isnan(units[u]["soc"])]
        m.BAT = pyo.Set(initialize=bats)
        d = lambda k: durations.get(k, 0.0)
        m.dur_up = pyo.Constraint(m.BAT, rule=lambda m, u: DT * m.p[u] + sum(d(k) * a(u, k) for k in UP)
                                  <= units[u]["soc"] - units[u]["soc_min"])
        m.dur_dn = pyo.Constraint(m.BAT, rule=lambda m, u: -DT * m.p[u] + d("REGDN") * a(u, "REGDN")
                                  <= units[u]["soc_max"] - units[u]["soc"])

    # Objective part: offer cost of awards minus the value of AS bought
    cost = sum(price[key] * m.x_as[key] for key in price)
    value = sum(seg[k, i][0] * m.y[k, i] + 0.5 * seg[k, i][1] * m.y[k, i] ** 2 for (k, i) in seg)
    m._as = {"a": a, "as_units": as_units, "K": K}
    return cost - value


def solve_interval(units, load, penalty, network=(), tee=False, asin=None, durations=None):
    """One SCED run. asin (from asbuild.build_as_interval) adds AS co-optimization; durations
    ({product: hours}) adds battery duration rows. Without asin the model is energy only."""

    segs = {}
    for u in units:
        curve = units[u]["curve"]
        for k, (a, b) in enumerate(itertools.pairwise(curve)):
            # a is [q_k, price_k], b is [q_k+1, price_k+1]
            # width = (b's MW) - (a's MW)
            width = b[0] - a[0]
            # if width == 0: skip this pair
            if width < 0.1:
                continue
            # slope = (b's price - a's price) / width
            slope = (b[1] - a[1]) / width
            price = a[1]
            if slope > 1000 or slope < 1e-3:
                price = (a[1] + b[1]) / 2
                slope = 0
            # segs[u, k] = (width, a's price, slope)
            segs[u, k] = (width, price, slope)

        # make an empty model
    m = pyo.ConcreteModel()
    # set U = unit names
    m.names = pyo.Set(initialize=units.keys())
    # set S = segment keys (u, k)
    m.S = pyo.Set(initialize=segs.keys(), dimen=2)
    # x[u,k] in [0, width]
    m.x = pyo.Var(m.S, bounds=lambda m, u, k: (0, segs[u, k][0]))
    # p[u] in [ldl, hdl]
    m.p = pyo.Var(m.names, bounds=lambda m, u: (units[u]["ldl"], units[u]["hdl"]))
    # s >= 0
    m.s = pyo.Var(within=pyo.NonNegativeReals)
    # link, one per unit: p[u] == curve start + sum of x[u,k] over that unit's segments.
    # Generator curves start at 0 MW; battery curves start at LDL (charging, negative MW)
    m.link = pyo.Constraint(m.names, rule=lambda m, u: m.p[u] == units[u]["curve"][0][0] + sum(m.x[v, k] for (v, k) in m.S if v == u))
    # balance: sum of p[u] + s == load
    m.balance = pyo.Constraint(expr=sum(m.p[u] for u in m.names) + m.s == load)
    # Network: for each constraint c, sum_u SF[c,u] * p[u] <= rhs[c] + v[c], with v[c] >= 0 priced at penalty[c]
    m.C = pyo.Set(initialize=range(len(network)))
    m.v = pyo.Var(m.C, within=pyo.NonNegativeReals)
    m.flow = pyo.Constraint(
        m.C,
        rule=lambda m, c: sum(sf * m.p[u] for u, sf in network[c]["sf"].items() if u in units)
        <= network[c]["rhs"] + m.v[c],
    )
    # AS co-optimization (optional): variables, rows and the AS part of the objective
    as_obj = _add_as(m, units, asin, durations) if asin is not None else 0
    # objective: sum over (u,k) of price*x + 0.5*slope*x**2, plus penalty*s, plus AS offers minus AS value
    m.cost = pyo.Objective(
        expr=sum(segs[u, k][1] * m.x[u, k] + 0.5 * segs[u, k][2] * m.x[u, k] ** 2 for (u, k) in m.S)
        + penalty * m.s
        + sum(network[c]["penalty"] * m.v[c] for c in m.C)
        + as_obj,
        sense=pyo.minimize,
    )
    # solve and get duals
    results = Highs().solve(
        m,
        tee=tee,
        solver_options={
            "primal_feasibility_tolerance": 1e-4,
            "dual_feasibility_tolerance": 1e-6,
            "time_limit": 20,
        },
    )
    duals = results.solution_loader.get_duals()

    out = {
        "dispatch": {u: pyo.value(m.p[u]) for u in m.names},
        "lambda": duals[m.balance],
        "shortage": pyo.value(m.s),
        "mu": [-duals[m.flow[c]] for c in m.C],
        "violation": [pyo.value(m.v[c]) for c in m.C],
    }
    if asin is not None:
        a, K = m._as["a"], m._as["K"]
        out["mcpc"] = {k: duals[m.as_balance[k]] for k in K if k in m.as_balance}
        out["awards"] = {u: {k: pyo.value(a(u, k)) for k in K} for u in m._as["as_units"]}
        out["cleared"] = {k: sum(pyo.value(m.y[k, i]) for i in range(len(asin["demand"][k]))) for k in K}
    return out