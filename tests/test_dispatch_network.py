from pytest import approx

from scedrep.dispatch import solve_interval

# Two units: A is cheap ($20, up to 150 MW), B is expensive ($50); load is 100 MW
UNITS = {
    "A": {"curve": [(0, 20), (150, 20)], "ldl": 0, "hdl": 150},
    "B": {"curve": [(0, 50), (100, 50)], "ldl": 0, "hdl": 100},
}


def test_line_limit_binds():
    # One line carries all of A's output (SF_A = 1, SF_B = 0) and is limited to 60 MW
    net = [{"sf": {"A": 1.0, "B": 0.0}, "rhs": 60.0, "penalty": 1000.0}]
    out = solve_interval(UNITS, 100, 5000, net)

    # A is capped at 60, B covers the rest and sets lambda
    assert out["dispatch"]["A"] == approx(60, abs=1e-3)
    assert out["dispatch"]["B"] == approx(40, abs=1e-3)
    assert out["lambda"] == approx(50, abs=1e-3)

    # Stationarity at A: 20 - lambda + mu * SF_A = 0, so mu = 30
    assert out["mu"][0] == approx(30, abs=1e-3)
    assert out["violation"][0] == approx(0, abs=1e-3)


def test_penalty_caps_shadow_price():
    # Same line, but violating it costs only $10/MW: cheaper than redispatching ($30/MW)
    net = [{"sf": {"A": 1.0, "B": 0.0}, "rhs": 60.0, "penalty": 10.0}]
    out = solve_interval(UNITS, 100, 5000, net)

    # The model overloads the line by 40 MW; mu hits its cap of 10
    assert out["dispatch"]["A"] == approx(100, abs=1e-3)
    assert out["violation"][0] == approx(40, abs=1e-3)
    assert out["mu"][0] == approx(10, abs=1e-3)

    # LMP at A = lambda - mu * SF_A = 20, so lambda = 30
    assert out["lambda"] == approx(30, abs=1e-3)

# A battery: charges (negative MW) when the price is at or below $10, discharges when at or above $40
ESR = {"curve": [(-50, 10), (0, 10), (0, 40), (50, 40)], "ldl": -50, "hdl": 50}


def test_battery_charges_when_energy_is_cheap():
    # Gas at $5 with plenty of room: the battery charges fully, and gas serves load plus charging
    units = {"A": {"curve": [(0, 5), (200, 5)], "ldl": 0, "hdl": 200}, "E": ESR}
    out = solve_interval(units, 100, 5000)
    assert out["dispatch"]["E"] == approx(-50, abs=1e-3)
    assert out["dispatch"]["A"] == approx(150, abs=1e-3)
    assert out["lambda"] == approx(5, abs=1e-3)


def test_battery_idle_between_its_prices():
    # Gas at $20: above the battery's $10 bid, below its $40 offer, so it does nothing
    units = {"A": {"curve": [(0, 20), (200, 20)], "ldl": 0, "hdl": 200}, "E": ESR}
    out = solve_interval(units, 100, 5000)
    assert out["dispatch"]["E"] == approx(0, abs=1e-3)
    assert out["lambda"] == approx(20, abs=1e-3)


def test_battery_discharges_when_energy_is_expensive():
    # Gas at $50: the battery's $40 offer is cheaper, so it discharges fully
    units = {"A": {"curve": [(0, 50), (200, 50)], "ldl": 0, "hdl": 200}, "E": ESR}
    out = solve_interval(units, 100, 5000)
    assert out["dispatch"]["E"] == approx(50, abs=1e-3)
    assert out["dispatch"]["A"] == approx(50, abs=1e-3)
    assert out["lambda"] == approx(50, abs=1e-3)