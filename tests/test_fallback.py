import pytest
from pytest import approx

from scedrep import dispatch, qp
from scedrep.dispatch import SolveError, solve_with_fallback

# Section 1.3's market: coal ($20, 100 MW) and gas ($50, 100 MW), load 150 -> lambda = 50
UNITS = {
    "coal": {"curve": [[0, 20], [100, 20]], "ldl": 0, "hdl": 100},
    "gas": {"curve": [[0, 50], [100, 50]], "ldl": 0, "hdl": 100},
}


@pytest.fixture
def piqp_first(monkeypatch):
    """Start the chain with PIQP whatever backend the test run uses."""
    monkeypatch.setattr(dispatch, "BACKEND", "piqp")


def test_strict_piqp_after_default_fails(piqp_first, monkeypatch):
    real = qp.solve_interval_piqp

    def default_fails(*args, settings=None, **kwargs):
        if settings is None:                        # the default attempt
            raise SolveError("PIQP status Status.PIQP_PRIMAL_INFEASIBLE")
        return real(*args, settings=settings, **kwargs)

    monkeypatch.setattr(qp, "solve_interval_piqp", default_fails)
    out = solve_with_fallback(UNITS, 150, 5000)
    assert out["solver"] == "piqp-strict"
    assert out["lambda"] == approx(50, abs=1e-4)


def test_highs_after_both_piqp_attempts_fail(piqp_first, monkeypatch):
    def always_fails(*args, **kwargs):
        raise SolveError("PIQP status Status.PIQP_MAX_ITER_REACHED")

    monkeypatch.setattr(qp, "solve_interval_piqp", always_fails)
    out = solve_with_fallback(UNITS, 150, 5000)
    assert out["solver"] == "highs"
    assert out["lambda"] == approx(50, abs=1e-4)


def test_raises_when_every_solver_fails(piqp_first, monkeypatch):
    def always_fails(*args, **kwargs):
        raise SolveError("PIQP failed")

    class FailingHighs:
        def solve(self, *args, **kwargs):
            raise SolveError("HiGHS failed")

    monkeypatch.setattr(qp, "solve_interval_piqp", always_fails)
    monkeypatch.setattr(dispatch, "Highs", FailingHighs)
    with pytest.raises(SolveError, match="HiGHS failed"):   # the last error in the chain
        solve_with_fallback(UNITS, 150, 5000)