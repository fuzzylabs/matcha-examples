"""Test suite for the train step."""
import pytest
from steps.load_data_step import load_data
from steps.train_step import train

from surprise import SVD
from surprise.trainset import Trainset


@pytest.fixture
def data(data_parameters: dict) -> Trainset:
    """A fixture to get the data used in training the model.

    Args:
        data_parameters (dict): parameters for train test split

    Returns:
        Trainset: training data
    """
    trainset, _ = load_data.entrypoint(data_parameters)

    return trainset


def test_correct_type(data: Trainset):
    """Test whether the trained model has correct type.

    Args:
        data (Trainset): training data
    """
    trainset = data

    assert isinstance(train.entrypoint(trainset), SVD)