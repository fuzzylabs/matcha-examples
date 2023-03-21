import pytest 
from types import SimpleNamespace

@pytest.fixture
def data_parameters() -> dict:
    """Create a dictionary for parameters used in load_data step

    Returns:
        dict: dictionary containing parameters for the step
    """
    parameters = SimpleNamespace()
    parameters.test_size = 0.25

    return parameters