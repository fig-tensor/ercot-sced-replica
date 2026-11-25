import math

import pandas as pd
import pytest
from pytest import approx

from scedrep.dispatch import solve_interval

NAN = math.nan


def unit(price, hdl, hsl=None, lsl=0.0, ldl=0.0, soc=NAN, soc_min=NAN, soc_max=NAN, caps=None):
    """A unit with a flat energy offer `price` from ldl to hdl MW (hsl defaults to hdl). A unit with soc is a
    battery. caps = {product: telemetered AS capability, MW/min}; a product left out has none (no cap row)."""
    return {"curve": [(ldl, price), (hdl, price)], "ldl": ldl, "hdl": hdl, "bp": 0.0,
            "hsl": hdl if hsl is None else hsl, "lsl": lsl, "soc": soc, "soc_min": soc_min, "soc_max": soc_max,
            **{f"cap_{k}": (caps or {}).get(k, NAN) for k in ["REGUP", "REGDN", "ECRS", "NSPIN"]}}


def blocks(*rows):
    """AS offer blocks: rows of (unit, block, sub-product, MW, price), all from ONRES rows (REGDN from REGDN).
    An RRS sub-type (RRSPFR, RRSUFR, RRSFFR) belongs to product RRS; any other sub-product is its product."""
    return pd.DataFrame([{"Resource Name": u, "Offer Type": "REGDN" if s == "REGDN" else "ONRES", "block": b,
                          "mw": q, "product": "RRS" if s.startswith("RRS") else s, "sub": s, "price": p}
                         for u, b, s, q, p in rows])


def asin(block_rows, demand, fixed=None, pfr=None):
    """AS inputs in the shape build_as_interval returns. demand = {product: [(P, s, w), ...]};
    pfr = {"min": MW, "unit_max": MW} adds the PFR rows (None: none)."""
    return {"blocks": blocks(*block_rows), "fixed": fixed or {}, "demand": demand, "pfr": pfr}


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



def test_highs_options_override():
    # A caller-supplied HiGHS option reaches the solver: a 0-iteration QP limit stops it before the optimum
    units = {"A": unit(10, 100, hsl=200)}
    inp = asin([("A", 1, "REGUP", 40, 5)], {"REGUP": [(100, -1, 50)]})   # sloped demand: a real QP
    # HiGHS-specific: pinned to the HiGHS backend
    assert solve_interval(units, 50, 5000, asin=inp, backend="highs",
                          highs_options={"qp_regularization_value": 1e-6})["mcpc"]
    with pytest.raises(Exception):  # noqa: B017 - any solver failure shows the option was applied
        solve_interval(units, 50, 5000, asin=inp, backend="highs", highs_options={"qp_iteration_limit": 0})



def test_sustained_limit_below_dispatch_floor():
    # ERCOT data, TIMPP_AGR2 on 2026-09-16 08:55:18: LDL 6.0 > HSL 5.1 (derate), HDL 55.8. The capacity rows
    # use H = max(HSL, HDL) = 55.8 >= LDL, so the run stays feasible. A ($10) runs to HDL; RegUp from A would
    # cost $10 of displaced energy + $1, more than B's $2, so B ($20, marginal) supplies it: lambda 20, MCPC 2.
    units = {"A": unit(10, 55.8, hsl=5.1, lsl=5.1, ldl=6.0), "B": unit(20, 200)}
    out = solve_interval(units, 100, 5000, asin=asin(
        [("A", 1, "REGUP", 10, 1), ("B", 1, "REGUP", 20, 2)], {"REGUP": [(100, 0, 10)]}))
    assert out["dispatch"]["A"] == approx(55.8, abs=1e-4)
    assert out["dispatch"]["B"] == approx(44.2, abs=1e-4)
    assert out["awards"]["A"]["REGUP"] == approx(0, abs=1e-4)
    assert out["awards"]["B"]["REGUP"] == approx(10, abs=1e-4)
    assert out["lambda"] == approx(20, abs=1e-4)
    assert out["mcpc"]["REGUP"] == approx(2, abs=1e-4)


def test_capacity_row_uses_hdl_above_hsl():
    # ERCOT data, MDANP_CT1 on 2026-09-15 01:55:16: HSL 227.2, HDL 227.57 (HDL above HSL by 0.37 MW); ERCOT's
    # base point plus NSpin award came to 227.57. With H = max(HSL, HDL) the cheap unit A ($10) runs to HDL,
    # not HSL; B ($20) is marginal for energy and supplies the NSpin ($5 vs $10 + $1 from A): lambda 20, MCPC 5.
    units = {"A": unit(10, 227.57, hsl=227.2, lsl=150, ldl=187.57), "B": unit(20, 500)}
    out = solve_interval(units, 300, 5000, asin=asin(
        [("A", 1, "NSPIN", 10, 1), ("B", 1, "NSPIN", 20, 5)], {"NSPIN": [(100, 0, 1)]}))
    assert out["dispatch"]["A"] == approx(227.57, abs=1e-4)
    assert out["awards"]["B"]["NSPIN"] == approx(1, abs=1e-4)
    assert out["lambda"] == approx(20, abs=1e-4)
    assert out["mcpc"]["NSPIN"] == approx(5, abs=1e-4)

def test_capability_cap_nspin():
    # A's telemetered NSpin capability is 1 MW/min, so it can carry at most 30 min x 1 = 30 MW of NSpin, though
    # its $1 block has 50 MW. B ($5) supplies the other 10 MW of the 40 MW bought: MCPC 5. C ($10) serves load.
    units = {"A": unit(50, 100, caps={"NSPIN": 1.0}), "B": unit(50, 100, caps={"NSPIN": 10.0}), "C": unit(10, 500)}
    out = solve_interval(units, 200, 5000, asin=asin(
        [("A", 1, "NSPIN", 50, 1), ("B", 1, "NSPIN", 50, 5)], {"NSPIN": [(100, 0, 40)]}))
    assert out["awards"]["A"]["NSPIN"] == approx(30, abs=1e-4)
    assert out["awards"]["B"]["NSPIN"] == approx(10, abs=1e-4)
    assert out["mcpc"]["NSPIN"] == approx(5, abs=1e-4)
    assert out["lambda"] == approx(10, abs=1e-4)


def test_capability_cap_regup_generator():
    # The cap applies to generator RegUp too: capability 1 MW/min x 5 min = 5 MW from A ($1); B ($5) the other 5.
    units = {"A": unit(50, 100, caps={"REGUP": 1.0}), "B": unit(50, 100, caps={"REGUP": 10.0}), "C": unit(10, 500)}
    out = solve_interval(units, 200, 5000, asin=asin(
        [("A", 1, "REGUP", 50, 1), ("B", 1, "REGUP", 50, 5)], {"REGUP": [(100, 0, 10)]}))
    assert out["awards"]["A"]["REGUP"] == approx(5, abs=1e-4)
    assert out["awards"]["B"]["REGUP"] == approx(5, abs=1e-4)
    assert out["mcpc"]["REGUP"] == approx(5, abs=1e-4)


def test_capability_cap_battery_ecrs_and_missing():
    # Battery A: ECRS capability 2 MW/min -> at most 10 x 2 = 20 MW. B has no capability telemetered (NaN): no
    # cap row, so its $5 block supplies the other 10 MW of the 30 MW bought. MCPC 5.
    units = {"A": unit(50, 100, soc=50, soc_min=0, soc_max=100, caps={"ECRS": 2.0}), "B": unit(50, 100),
             "C": unit(10, 500)}
    out = solve_interval(units, 200, 5000, asin=asin(
        [("A", 1, "ECRS", 50, 1), ("B", 1, "ECRS", 50, 5)], {"ECRS": [(100, 0, 30)]}))
    assert out["awards"]["A"]["ECRS"] == approx(20, abs=1e-4)
    assert out["awards"]["B"]["ECRS"] == approx(10, abs=1e-4)
    assert out["mcpc"]["ECRS"] == approx(5, abs=1e-4)


def rrs_units():
    """A, B, C: AS sellers with expensive energy ($50); D ($10) serves the 200 MW load, so lambda = 10."""
    return {"A": unit(50, 100), "B": unit(50, 100), "C": unit(50, 100), "D": unit(10, 500)}


def test_pfr_floor_and_unit_cap():
    # RRS 40 MW bought. FFR from B at $1 is cheapest, but at least 30 MW must be PFR, at most 20 MW per unit:
    # A carries 20 PFR ($5), C the other 10 ($8), B the remaining 10 as FFR. One more MW of RRS would come from
    # B: MCPC = $1. (The floor's own price, $8 - $1 = $7, is not part of the RRS MCPC.)
    out = solve_interval(rrs_units(), 200, 5000, asin=asin(
        [("A", 1, "RRSPFR", 50, 5), ("B", 1, "RRSFFR", 50, 1), ("C", 1, "RRSPFR", 50, 8)],
        {"RRS": [(100, 0, 40)]}, pfr={"min": 30, "unit_max": 20}))
    aw = out["awards"]
    assert aw["A"]["RRSPFR"] == approx(20, abs=1e-4)
    assert aw["C"]["RRSPFR"] == approx(10, abs=1e-4)
    assert aw["B"]["RRS"] == approx(10, abs=1e-4)
    assert aw["B"]["RRSPFR"] == approx(0, abs=1e-4)
    assert out["mcpc"]["RRS"] == approx(1, abs=1e-4)
    assert out["pfr_short"] == approx(0, abs=1e-4)
    assert out["lambda"] == approx(10, abs=1e-4)


def test_pfr_floor_counts_fixed_supply():
    # 25 MW of PFR comes from resources the model doesn't dispatch (fixed), so only 5 more are needed:
    # A 5 PFR ($5), B 15 FFR ($1) for the 20 MW of the 45 MW curve not already covered. MCPC $1.
    out = solve_interval(rrs_units(), 200, 5000, asin=asin(
        [("A", 1, "RRSPFR", 50, 5), ("B", 1, "RRSFFR", 50, 1)],
        {"RRS": [(100, 0, 45)]}, fixed={"RRS": 25, "RRSPFR": 25}, pfr={"min": 30, "unit_max": 20}))
    assert out["awards"]["A"]["RRSPFR"] == approx(5, abs=1e-4)
    assert out["awards"]["B"]["RRS"] == approx(15, abs=1e-4)
    assert out["mcpc"]["RRS"] == approx(1, abs=1e-4)


def test_pfr_floor_shortfall():
    # Only A offers PFR and may carry at most 20 MW: the floor (30) is 10 MW short, priced at the penalty
    # rather than making the run infeasible. B fills the rest of the 50 MW as FFR.
    out = solve_interval(rrs_units(), 200, 5000, asin=asin(
        [("A", 1, "RRSPFR", 50, 5), ("B", 1, "RRSFFR", 50, 1)],
        {"RRS": [(100, 0, 50)]}, pfr={"min": 30, "unit_max": 20}))
    assert out["awards"]["A"]["RRSPFR"] == approx(20, abs=1e-4)
    assert out["pfr_short"] == approx(10, abs=1e-4)
    assert out["awards"]["B"]["RRS"] == approx(30, abs=1e-4)


def test_no_ufr_from_dispatched_units():
    # UFR is a Load Resource service: A's $1 UFR price is dropped (A offers nothing else, so it gets no AS
    # variables at all) and B's FFR at $5 supplies RRS. MCPC $5.
    out = solve_interval(rrs_units(), 200, 5000, asin=asin(
        [("A", 1, "RRSUFR", 50, 1), ("B", 1, "RRSFFR", 50, 5)], {"RRS": [(100, 0, 10)]}))
    assert "A" not in out["awards"]
    assert out["awards"]["B"]["RRS"] == approx(10, abs=1e-4)
    assert out["mcpc"]["RRS"] == approx(5, abs=1e-4)


def test_pricing_run_breaks_the_floor_tie():
    # The tie: RRS curve 40 MW, 10 MW fixed, so 30 MW of room = the 30 MW PFR floor. Quantities come from the
    # dispatch run (A 20 PFR at its cap, C 10 PFR, B no FFR); the price from the pricing run without the PFR rows,
    # where B's FFR at $1 is the marginal RRS offer: $1. The dispatch run's own RRS price is not unique here
    # (PIQP lands far below $1), which is why prices come from the pricing run.
    from scedrep.dispatch import solve_with_pricing

    inp = asin([("A", 1, "RRSPFR", 50, 5), ("B", 1, "RRSFFR", 50, 1), ("C", 1, "RRSPFR", 50, 8)],
               {"RRS": [(100, 0, 40)]}, fixed={"RRS": 10, "RRSPFR": 0}, pfr={"min": 30, "unit_max": 20})
    out = solve_with_pricing(rrs_units(), 200, 5000, asin=inp)
    assert out["awards"]["A"]["RRSPFR"] == approx(20, abs=1e-4)
    assert out["awards"]["C"]["RRSPFR"] == approx(10, abs=1e-4)
    assert out["awards"]["B"]["RRS"] == approx(0, abs=1e-4)
    assert out["pfr_short"] == approx(0, abs=1e-4)
    assert out["pricing"]["mcpc"]["RRS"] == approx(1, abs=1e-4)
    assert out["pricing"]["lambda"] == approx(10, abs=1e-4)


def test_no_pricing_run_without_pfr_rows():
    from scedrep.dispatch import solve_with_pricing

    units = {"A": unit(10, 100), "B": unit(20, 100)}
    out = solve_with_pricing(units, 50, 5000, asin=asin([("A", 1, "REGUP", 30, 5)], {"REGUP": [(100, 0, 10)]}))
    assert "pricing" not in out
