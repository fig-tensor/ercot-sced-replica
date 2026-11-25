import pytest

from scedrep import dispatch


@pytest.fixture(autouse=True, params=["highs", "piqp"])
def backend(request, monkeypatch):
    """Run every test once per solver backend: both must give the same known answers."""
    monkeypatch.setattr(dispatch, "BACKEND", request.param)
    return request.param