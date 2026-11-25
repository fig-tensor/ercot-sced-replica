from pytest import approx

from scedrep.dispatch import solve_interval

# Set tolerance level to thousandth MW or tenth of cent per MWh for asserts
TOL = 1e-3


def test_interior_lambda():
    # units are in MW and $
    units = {
        "coal": {"curve": [[0, 20], [100, 20]], "ldl": 0, "hdl": 100},
        "gas": {"curve": [[0, 50], [100, 50]], "ldl": 0, "hdl": 100},
    }
    # load is in MW
    load = 150
    # solve func
    result = solve_interval(units, load, penalty=5000)

    assert result["dispatch"]["coal"] == approx(100)
    assert result["dispatch"]["gas"] == approx(50)
    assert result["lambda"] == approx(50)


def test_boundary_lambda():
    # units are in MW and $
    units = {
        "coal": {"curve": [[0, 20], [100, 20]], "ldl": 0, "hdl": 100},
        "gas": {"curve": [[0, 50], [100, 50]], "ldl": 0, "hdl": 100},
    }
    # load is in MW
    load = 100
    # olve func
    result = solve_interval(units, load, penalty=5000)

    assert result["dispatch"]["coal"] == approx(100)
    assert result["dispatch"]["gas"] == approx(0, abs=1e-6)  # interior-point solvers stop a hair inside bounds
    assert 20 <= result["lambda"] <= 50


def test_shortage():
    # units are in MW and $
    units = {
        "coal": {"curve": [[0, 20], [100, 20]], "ldl": 0, "hdl": 100},
        "gas": {"curve": [[0, 50], [100, 50]], "ldl": 0, "hdl": 100},
    }
    # load is in MW
    load = 250
    # solve func
    result = solve_interval(units, load, penalty=5000)

    assert result["dispatch"]["coal"] == approx(100)
    assert result["dispatch"]["gas"] == approx(100)
    assert result["lambda"] == approx(5000)
    assert result["shortage"] == approx(50)


def test_sloped_curves():
    # units are in MW and $
    units = {
        "coal": {"curve": [[0, 10], [500, 60]], "ldl": 0, "hdl": 500},
        "gas": {"curve": [[0, 20], [500, 120]], "ldl": 0, "hdl": 500},
    }
    # load is in MW
    load = 200
    # solve func
    result = solve_interval(units, load, penalty=5000)

    assert result["dispatch"]["coal"] == approx(500 / 3, abs=TOL)
    assert result["dispatch"]["gas"] == approx(100 / 3, abs=TOL)
    assert result["lambda"] == approx(80 / 3, abs=TOL)
