import itertools
import math
from collections import defaultdict

import pyomo.environ as pyo
from pyomo.contrib.solver.common.util import NoDualsError, NoOptimalSolutionError
from pyomo.contrib.solver.solvers.highs import Highs

DT = 5 / 60  # SCED interval length in hours
UP = ("REGUP", "RRS", "ECRS", "NSPIN")  # up-products: share headroom above p
RAMP = 5 / 7  # regulation ramp factor in the ramp rows (ERCOT training material)
# Capability caps: award of product k <= RAMP_MINUTES[k] x the unit's telemetered AS capability for k (MW/min).
# Scored against ERCOT's own awards (Sept 15-21, ~650,000 awards): never broken for any product or resource type,
# binding tens of thousands of times; minutes x disclosed ramp rate was broken thousands of times.
RAMP_MINUTES = {"REGUP": 5, "REGDN": 5, "ECRS": 10, "NSPIN": 30}
# RRS sub-types the model can award: PFR and FFR. UFR (under-frequency relay) is a Load Resource service; the
# model doesn't dispatch Load Resources (their awards are fixed supply), and generators and batteries got no
# UFR award in Sept 15-21.
RRS_MODEL_SUB = ("RRSPFR", "RRSFFR")
PFR_PENALTY = 10_000.0  # $/MW short of the PFR floor. ERCOT doesn't publish one; above every AS curve cap
BACKEND = "piqp"  # default solver for solve_interval: "piqp" (scedrep.qp, interior point) or "highs" (Pyomo + HiGHS)
PIQP_STRICT = {"max_iter": 1000, "eps_primal_inf": 1e-8}  # more iterations, stricter infeasibility detection


class SolveError(RuntimeError):
    """A solver backend did not return an optimal solution."""


def energy_segments(units):
    """Offer-curve segments per unit, cleaned for numerical conditioning.

    Returns {(u, k): (width, price, slope)}. Segments narrower than 0.1 MW are dropped; slopes above 1000 or
    below 1e-3 are flattened at the segment's average price. Shared by both backends, so both solve the
    same model.
    """
    segs = {}
    for u in units:
        curve = units[u]["curve"]
        for k, (a, b) in enumerate(itertools.pairwise(curve)):
            # a is [q_k, price_k], b is [q_k+1, price_k+1]
            width = b[0] - a[0]
            if width < 0.1:
                continue
            slope = (b[1] - a[1]) / width
            price = a[1]
            if slope > 1000 or slope < 1e-3:
                price = (a[1] + b[1]) / 2
                slope = 0
            segs[u, k] = (width, price, slope)
    return segs


def as_terms(units, asin):
    """AS model data shared by both backends, from asbuild.build_as_interval output.

    Returns a dict:
      K         products with a demand curve this run
      price     {(u, o, b, s): $/MW-h}   offered (unit, offer type, block, sub-product s) of dispatched units;
                                         s = product, except RRS: RRSPFR or RRSFFR
      Q         {(u, o, b): MW}          shared MW of each block
      by_award  {(u, k): [price keys]}   blocks offering product k (all its sub-types) per unit; also
                                         (u, "RRSPFR") for PFR alone
      by_block  {(u, o, b): [keys]}      products offered by each block
      as_units  units that offer AS (they get capacity and ramp rows)
      hsl, lsl  {u: MW}                  limits for the capacity rows (see below)
      award_cap {(u, k): MW}             award of k to u <= this (capability caps; PFR per-resource cap)
      pfr       {"min", "unit_max"} or None   RRS-PFR floor and per-resource cap (None: no PFR rows)
      seg       {(k, i): (P, s, w)}      demand segments
      fixed     {k: MW}                  F_k
    """
    demand = asin["demand"]
    K = list(demand)
    blk = asin["blocks"]
    if "sub" not in blk.columns:
        blk = blk.assign(sub=blk["product"])  # inputs without sub-types: one sub per product
    keep = blk["Resource Name"].isin(units.keys()) & blk["product"].isin(K)
    keep &= (blk["product"] != "RRS") | blk["sub"].isin(RRS_MODEL_SUB)  # no UFR for dispatched units
    blk = blk[keep]
    cols = [blk["Resource Name"], blk["Offer Type"], blk["block"], blk["sub"]]
    price = {(u, o, int(b), s): float(pr) for u, o, b, s, pr in zip(*cols, blk["price"], strict=True)}
    Q = {(u, o, int(b)): float(q) for u, o, b, q in zip(*cols[:3], blk["mw"], strict=True)}
    by_award, by_block = defaultdict(list), defaultdict(list)
    for key in price:
        u, s = key[0], key[3]
        by_award[u, "RRS" if s in RRS_MODEL_SUB else s].append(key)  # sub-types add up to RRS
        if s == "RRSPFR":
            by_award[u, "RRSPFR"].append(key)
        by_block[key[:3]].append(key)
    as_units = sorted({key[0] for key in price})
    for u in as_units:
        if math.isnan(units[u]["hsl"]) or math.isnan(units[u]["lsl"]):
            raise ValueError(f"unit {u} offers AS but has no HSL/LSL")
    # Limits for the capacity rows: H = max(HSL, HDL), L = min(LSL, HDL).
    # ERCOT dispatches up to HDL when HDL is above HSL (Sept 15-21: 4,165 unit-runs with HDL > HSL, median gap
    # 0.3 MW; base point above max(HSL, LDL) in 3,972), and its own base points plus up-awards never exceed
    # max(HSL, HDL), while they exceed max(HSL, LDL) 19 times (by 0.1-0.8 MW). H >= HDL > LDL for every
    # dispatched unit, so the rows stay feasible when a derate puts HSL below LDL (TIMPP_AGR2, 2026-09-16).
    hsl = {u: max(units[u]["hsl"], units[u]["hdl"]) for u in as_units}
    lsl = {u: min(units[u]["lsl"], units[u]["hdl"]) for u in as_units}
    # Award caps a[u,k] <= cap. Capability caps (see RAMP_MINUTES), for every unit; no row where the unit doesn't
    # offer the product or has no telemetered capability (NaN). A capability of 0 caps the award at 0.
    award_cap = {}
    for u in as_units:
        for k, minutes in RAMP_MINUTES.items():
            cap = units[u].get(f"cap_{k}", math.nan)
            if (u, k) in by_award and cap >= 0:  # NaN >= 0 is False
                award_cap[u, k] = minutes * cap
    # PFR per-resource cap
    pfr = asin.get("pfr") if "RRS" in K else None
    if pfr:
        for u in as_units:
            if (u, "RRSPFR") in by_award:
                award_cap[u, "RRSPFR"] = pfr["unit_max"]
    seg = {(k, i): sw for k in K for i, sw in enumerate(demand[k])}
    return {"K": K, "price": price, "Q": Q, "by_award": by_award, "by_block": by_block,
            "as_units": as_units, "hsl": hsl, "lsl": lsl, "award_cap": award_cap, "pfr": pfr, "seg": seg,
            "fixed": asin.get("fixed", {})}



def _add_as(m, units, asin, durations):
    """Add ancillary-service (AS) variables and rows to model m; return the AS part of the objective.

    asin: output of asbuild.build_as_interval -> {"blocks", "fixed", "demand"}.
    Variables: x[u,o,b,k] >= 0, MW of offer block (u, o, b) awarded for product k; y[k,i] in [0, w],
    MW filled on demand segment i of product k. Awards a[u,k] = sum of x are expressions.
    Rows: AS balance per product (dual = MCPC), block capacity, capacity and ramp rows for units that
    offer AS, award caps (minutes x capability, PFR per resource), the PFR floor, and battery duration rows if
    durations = {product: hours} is given.
    """
    t = as_terms(units, asin)
    K, price, Q, seg, fixed = t["K"], t["price"], t["Q"], t["seg"], t["fixed"]
    by_award, by_block, as_units = t["by_award"], t["by_block"], t["as_units"]
    hsl, lsl = t["hsl"], t["lsl"]
    demand = asin["demand"]

    m.X = pyo.Set(initialize=list(price), dimen=4)
    m.x_as = pyo.Var(m.X, within=pyo.NonNegativeReals)

    def a(u, k):
        """Award of product k to unit u: sum of x over its blocks offering k (0 if none)."""
        return sum(m.x_as[key] for key in by_award.get((u, k), []))

    # Block capacity: the products a block prices share its MW
    m.BLK = pyo.Set(initialize=list(by_block), dimen=3)
    m.block_cap = pyo.Constraint(m.BLK, rule=lambda m, u, o, b: sum(m.x_as[key] for key in by_block[u, o, b]) <= Q[u, o, b])

    # Demand segments (P, s, w): fill y in [0, w], worth P*y + s*y^2/2 (s <= 0: concave)
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
    m.cap_hi = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] + up(u) <= hsl[u])
    m.cap_lo = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] - a(u, "REGDN") >= lsl[u])
    m.ramp_hi = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] + RAMP * a(u, "REGUP") <= units[u]["hdl"])
    m.ramp_lo = pyo.Constraint(m.AU, rule=lambda m, u: m.p[u] - RAMP * a(u, "REGDN") >= units[u]["ldl"])
    # Award caps: a[u,k] <= minutes_k x capability; PFR per resource
    award_cap = t["award_cap"]
    m.CAPK = pyo.Set(initialize=list(award_cap), dimen=2)
    m.award_cap = pyo.Constraint(m.CAPK, rule=lambda m, u, k: a(u, k) <= award_cap[u, k])

    # PFR floor: sum_u PFR awards + fixed PFR + shortfall >= minimum, shortfall priced at PFR_PENALTY
    pfr_obj = 0
    if t["pfr"]:
        m.pfr_short = pyo.Var(within=pyo.NonNegativeReals)
        m.pfr_floor = pyo.Constraint(expr=sum(a(u, "RRSPFR") for u in as_units) + fixed.get("RRSPFR", 0.0)
                                     + m.pfr_short >= t["pfr"]["min"])
        pfr_obj = PFR_PENALTY * m.pfr_short

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
    m._as = {"a": a, "as_units": as_units, "K": K, "pfr": bool(t["pfr"])}
    return cost - value + pfr_obj


def solve_interval(units, load, penalty, network=(), tee=False, asin=None, durations=None, highs_options=None,
                   backend=None, piqp_settings=None):
    """One SCED run. asin (from asbuild.build_as_interval) adds AS co-optimization; durations
    ({product: hours}) adds battery duration rows. Without asin the model is energy only.
    backend: "highs" (Pyomo + HiGHS) or "piqp" (scedrep.qp); None uses the module setting BACKEND.
    highs_options: HiGHS settings that override the defaults below (e.g. qp_regularization_value);
    piqp_settings: PIQP settings (e.g. PIQP_STRICT)."""
    backend = backend or BACKEND
    if backend == "piqp":
        # imported here, not at the top: qp imports from this module
        from scedrep.qp import solve_interval_piqp
        return solve_interval_piqp(units, load, penalty, network, verbose=tee, asin=asin, durations=durations, settings=piqp_settings)
    if backend != "highs":
        raise ValueError(f"unknown backend {backend!r}")

    segs = energy_segments(units)

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
    options = {
        "primal_feasibility_tolerance": 1e-4,
        "dual_feasibility_tolerance": 1e-6,
        "time_limit": 20,
        **(highs_options or {}),  # caller's settings win
    }
    results = Highs().solve(m, tee=tee, solver_options=options)
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
        ks = K + ["RRSPFR"] if m._as["pfr"] else K  # PFR awards reported alongside the products
        out["awards"] = {u: {k: pyo.value(a(u, k)) for k in ks} for u in m._as["as_units"]}
        out["cleared"] = {k: sum(pyo.value(m.y[k, i]) for i in range(len(asin["demand"][k]))) for k in K}
        if m._as["pfr"]:
            out["pfr_short"] = pyo.value(m.pfr_short)  # MW short of the PFR floor
    return out



def solve_with_fallback(units, load, penalty, network=(), asin=None, durations=None):
    """Solve one run, falling back to other solvers if one fails; the output records which one solved it.

    Order, starting from the default backend: PIQP; PIQP with stricter settings (more iterations, stricter
    infeasibility detection: PIQP can stop early or declare a feasible problem infeasible); HiGHS, a different
    algorithm (active set) that does not share PIQP's failure modes. Raises the last error if all fail.
    """
    piqp = [("piqp", {"backend": "piqp"}), ("piqp-strict", {"backend": "piqp", "piqp_settings": PIQP_STRICT})]
    highs = [("highs", {"backend": "highs"})]
    chain = piqp + highs if BACKEND == "piqp" else highs + piqp
    error = None
    for label, kw in chain:
        try:
            out = solve_interval(units, load, penalty, network, asin=asin, durations=durations, **kw)
            out["solver"] = label
            return out
        except (NoOptimalSolutionError, NoDualsError, SolveError) as e:
            error = e
    raise error


def solve_with_pricing(units, load, penalty, network=(), asin=None, durations=None):
    """A dispatch run and, when the RRS-PFR rows are in, a separate pricing run without them.

    ERCOT's awards obey the PFR floor exactly (total PFR = 1,377 MW in nearly every run, Sept 15-21), yet its
    RRS price matches a model run without the floor (Sept 15 and 18: median |error| $0.004, no run off by more
    than $1), where the dispatch run's price can carry the floor's dual (down to -$10,000 when the floor and the
    RRS curve's end tie). So: quantities (dispatch, awards) from the dispatch run with the PFR rows; prices from a
    pricing run without them, returned as out["pricing"] = {"lambda", "mu", "mcpc", "solver"}. The dispatch
    run's own prices stay in out. Both runs go through solve_with_fallback.
    """
    out = solve_with_fallback(units, load, penalty, network, asin=asin, durations=durations)
    if asin is not None and asin.get("pfr"):
        pr = solve_with_fallback(units, load, penalty, network, asin={**asin, "pfr": None}, durations=durations)
        out["pricing"] = {"lambda": pr["lambda"], "mu": pr["mu"], "mcpc": pr["mcpc"], "solver": pr["solver"]}
    return out
