import pyomo.environ as pyo


def solve_interval(units, load):
    # make an empty model
    m = pyo.ConcreteModel()
    # set names = names of the units
    m.names = pyo.Set(initialize=units.keys())
    # variables p[u] >= 0
    m.p = pyo.Var(m.names, within=pyo.NonNegativeReals)
    # constraint for each u: p[u] <= max MW of u
    m.cap = pyo.Constraint(m.names, rule=lambda m, u: m.p[u] <= units[u][0])
    # constraint: sum of p[u] == load
    m.balance = pyo.Constraint(expr=sum(m.p[u] for u in m.names) == load)
    # objective: minimize sum of price[u] * p[u]
    m.cost = pyo.Objective(
        expr=sum(units[u][1] * m.p[u] for u in m.names), sense=pyo.minimize
    )
    # ask for duals
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # solve
    pyo.SolverFactory("appsi_highs").solve(m)
    # return a dict: {"dispatch": {u: p[u] value}, "lambda": dual of the balance constraint}

    return {
        "dispatch": {u: pyo.value(m.p[u]) for u in m.names},
        "lambda": m.dual[m.balance],
    }
