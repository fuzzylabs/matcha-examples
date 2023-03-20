import pytest 
from types import SimpleNamespace

@pytest.fixture
def data_parameters():
    parameters = SimpleNamespace()
    parameters.test_size = 0.25
    return parameters