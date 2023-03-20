import pytest

from steps import (
    load_data,
    train,
)

from surprise import SVD


@pytest.fixture
def data(data_parameters):
    trainset, _ = load_data.entrypoint(data_parameters)

    return trainset


def test_correct_type(data):
    trainset = data

    assert isinstance(train.entrypoint(trainset), 
                      SVD)