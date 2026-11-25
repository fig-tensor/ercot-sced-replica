import math

import pandas as pd
import pytest
from pytest import approx

from scedrep.dispatch import solve_interval

NAN = math.nan


def unit(price, hdl, hsl=None, lsl=0.0, ldl=0.0, soc=NAN, soc_min=NAN, soc_max=NAN):
    """A unit with a flat energy offer `price` from ldl to hdl MW (hsl defaults to hdl)."""
    return {"curve": [(ldl, price), (hdl, price)], "ldl": ldl, "hdl": hdl, "bp": 0.0,
            "hsl": hdl if hsl is None else hsl, "lsl": lsl, "soc": soc, "soc_min": soc_min, "soc_max": soc_max}


def blocks(*rows):
    """AS offer blocks: rows of (unit, block, product, MW, price), all from ONRES rows (REGDN from REGDN)."""
    return pd.DataFrame([{"Resource Name": u, "Offer Type": "REGDN" if k == "REGDN" else "ONRES", "block": b,
                          "mw": q, "product": k, "price": p} for u, b, k, q, p in rows])


def asin(block_rows, demand, fixed=None):
    """AS inputs in the shape build_as_interval returns. demand = {product: [(P, s, w), ...]}."""
    return {"blocks": blocks(*block_rows), "fixed": fixed or {}, "demand": demand}


def test_interior_as_price():
    # Energy: A ($10) is marginal at load 50. RegUp: A 30 MW at $5, B 40 MW at $10, demand $100 for 50 MW.
    units = {"A": unit(10, 100, hsl=200), "B": unit(20, 100)}
    out = solve_interval(units, 50, 5000, asin=asin(
        [("A", 1, "REGUP", 30, 5), ("B", 1, "REGUP", 40, 10)], {"REGUP": [(100, 0, 50)]}))
    assert out["lambda"] == approx(10)
    assert out["awards"]["A"]["REGUP"] == approx(30)
    assert out["awards"]["B"]["REGUP"] == approx(20)       # B is inside its block...
    assert out["mcpc"]["REGUP"] == approx(10)              # ...so it sets the price


def test_curve_end_bought_price_from_supply():
    # Demand $100 for 30 MW; A offers 40 MW at $5 and is partly awarded: MCPC = $5, not $100
    units = {"A": unit(10, 100, hsl=200)}
    out = solve_interval(units, 50, 5000, asin=asin([("A", 1, "REGUP", 40, 5)], {"REGUP": [(100, 0, 30)]}))
    assert out["cleared"]["REGUP"] == approx(30)
    assert out["mcpc"]["REGUP"] == approx(5)


def test_degenerate_curve_end():
    # A (30 MW at $5) exactly fills the 30 MW curve, B ($10) gets nothing: any MCPC in [5, 10] is valid
    units = {"A": unit(10, 100, hsl=200), "B": unit(20, 100)}
    out = solve_interval(units, 50, 5000, asin=asin(
        [("A", 1, "REGUP", 30, 5), ("B", 1, "REGUP", 40, 10)], {"REGUP": [(100, 0, 30)]}))
    assert out["awards"]["A"]["REGUP"] == approx(30)
    assert 5 - 1e-6 <= out["mcpc"]["REGUP"] <= 10 + 1e-6


def test_opportunity_cost():
    # A ($20) is at HSL = 100 serving energy; B ($50) is marginal, lambda = 50. Only A offers RegUp ($1).
    # Each MW of RegUp moves 1 MW of energy from A to B: MCPC = 1 + (50 - 20) = 31.
    units = {"A": unit(20, 100), "B": unit(50, 200)}
    out = solve_interval(units, 150, 5000, asin=asin([("A", 1, "REGUP", 50, 1)], {"REGUP": [(1000, 0, 10)]}))
    assert out["lambda"] == approx(50)
    assert out["dispatch"]["A"] == approx(90)
    assert out["dispatch"]["B"] == approx(60)
    assert out["mcpc"]["REGUP"] == approx(31)


def test_shared_block_capacity():
    # One 20 MW block priced for RegUp ($1) and ECRS ($2); demand 15 MW of each at $100.
    # Shared: 20 MW total. RegUp (cheaper) gets its 15 MW, ECRS the remaining 5 and is short, so ECRS is
    # priced on its demand curve ($100). The full block is worth 100 - 2 = $98 per MW, which RegUp pays as
    # opportunity cost: MCPC_RegUp = 1 + 98 = $99.
    units = {"A": unit(10, 100, hsl=500)}
    out = solve_interval(units, 50, 5000, asin=asin(
        [("A", 1, "REGUP", 20, 1), ("A", 1, "ECRS", 20, 2)],
        {"REGUP": [(100, 0, 15)], "ECRS": [(100, 0, 15)]}))
    assert out["awards"]["A"]["REGUP"] == approx(15)
    assert out["awards"]["A"]["ECRS"] == approx(5)
    assert out["mcpc"]["ECRS"] == approx(100)
    assert out["mcpc"]["REGUP"] == approx(99)


def test_battery_duration():
    # Battery: 10 MWh stored, offers NSpin 40 MW at $1; demand $100 for 60 MW. With d_NSPIN = 4 h and the
    # battery idle (its $40 energy offer is above lambda = $10): 4a <= 10, so a = 2.5 and MCPC = $100.
    units = {"G": unit(10, 100), "BAT": unit(40, 50, hsl=50, soc=10, soc_min=0, soc_max=100)}
    inp = asin([("BAT", 1, "NSPIN", 40, 1)], {"NSPIN": [(100, 0, 60)]})
    out = solve_interval(units, 20, 5000, asin=inp, durations={"NSPIN": 4})
    assert out["awards"]["BAT"]["NSPIN"] == approx(2.5)
    assert out["mcpc"]["NSPIN"] == approx(100)
    out = solve_interval(units, 20, 5000, asin=inp)        # durations off: the whole block
    assert out["awards"]["BAT"]["NSPIN"] == approx(40)


def test_fixed_supply_and_no_asin():
    # 25 MW of RegUp fixed elsewhere; demand $100 for 50 MW; A offers 40 at $5 -> A gets 25, MCPC = 5
    units = {"A": unit(10, 100, hsl=200)}
    out = solve_interval(units, 50, 5000, asin=asin([("A", 1, "REGUP", 40, 5)], {"REGUP": [(100, 0, 50)]},
                                                    fixed={"REGUP": 25}))
    assert out["awards"]["A"]["REGUP"] == approx(25)
    assert out["mcpc"]["REGUP"] == approx(5)
    assert "mcpc" not in solve_interval(units, 50, 5000)  # energy only: unchanged output


def test_as_unit_without_hsl_raises():
    units = {"A": unit(10, 100, hsl=NAN)}
    with pytest.raises(ValueError, match="HSL"):
        solve_interval(units, 50, 5000, asin=asin([("A", 1, "REGUP", 40, 5)], {"REGUP": [(100, 0, 50)]}))