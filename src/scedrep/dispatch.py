import itertools

import pyomo.environ as pyo
from pyomo.contrib.solver.solvers.highs import Highs


def solve_interval(units, load, penalty):

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
    # link, one per unit: p[u] == sum of x[u,k] over that unit's segments
    m.link = pyo.Constraint(
        m.names, rule=lambda m, u: m.p[u] == sum(m.x[v, k] for (v, k) in m.S if v == u)
    )
    # balance: sum of p[u] + s == load
    m.balance = pyo.Constraint(expr=sum(m.p[u] for u in m.names) + m.s == load)
    # objective: sum over (u,k) of price*x + 0.5*slope*x**2, plus penalty*s
    m.cost = pyo.Objective(
        expr=sum(
            segs[u, k][1] * m.x[u, k] + 0.5 * segs[u, k][2] * m.x[u, k] ** 2
            for (u, k) in m.S
        )
        + penalty * m.s,
        sense=pyo.minimize,
    )
    # solve and get duals
    results = Highs().solve(
        m,
        solver_options={
            "primal_feasibility_tolerance": 1e-6,
            "dual_feasibility_tolerance": 1e-6,
        },
    )
    duals = results.solution_loader.get_duals()

    return {
        "dispatch": {u: pyo.value(m.p[u]) for u in m.names},
        "lambda": duals[m.balance],
        "shortage": pyo.value(m.s),
    }
