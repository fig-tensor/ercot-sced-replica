from pytest import approx

from scedrep.dispatch import solve_interval


def test_interior_lambda():
    # units are in MW and $
    units = {"coal": (100, 20), "gas": (100, 50)}
    # load is in MW
    load = 150
    # solve func
    result = solve_interval(units, load)

    assert result["dispatch"]["coal"] == approx(100)
    assert result["dispatch"]["gas"] == approx(50)
    assert result["lambda"] == approx(50)


def test_boundary_lambda():
    # units are in MW and $
    units = {"coal": (100, 20), "gas": (100, 50)}
    # load is in MW
    load = 100
    # olve func
    result = solve_interval(units, load)

    assert result["dispatch"]["coal"] == approx(100)
    assert result["dispatch"]["gas"] == approx(0)
    assert 20 <= result["lambda"] <= 50
