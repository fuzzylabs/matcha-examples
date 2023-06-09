"""Test suite for the load_data step."""
from steps.load_data_step import load_data

from surprise.trainset import Trainset


EXPECTED_DATA_LENGTH = 100000


def test_load_data_step_expected_output_types(data_parameters: dict):
    """Test whether the load_data step output the expected types.

    Args:
        data_parameters (dict): parameters for train test split
    """
    trainset, testset = load_data.entrypoint(data_parameters)

    assert isinstance(trainset, Trainset)
    assert isinstance(testset, list)


def test_load_data_step_expected_data_amount(data_parameters: dict):
    """Test whether the load_data step split train and test data as expected.

    Args:
        data_parameters (dict): parameters for train test split
    """
    trainset, testset = load_data.entrypoint(data_parameters)

    expected_size_train = int(EXPECTED_DATA_LENGTH * (1 - data_parameters.test_size))
    expected_size_test = int(EXPECTED_DATA_LENGTH * data_parameters.test_size)

    assert trainset.n_ratings + len(testset) == EXPECTED_DATA_LENGTH

    assert trainset.n_ratings == expected_size_train
    assert len(testset) == expected_size_test

