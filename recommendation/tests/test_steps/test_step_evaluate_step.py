import pytest

from steps import (
    load_data,
    train,
    evaluate
)


BENCHMARK_SVD_SCORE = 0.934

@pytest.fixture
def data(data_parameters):
    trainset, testset = load_data.entrypoint(data_parameters)

    return trainset, testset


@pytest.fixture
def model(data):
    trainset, _ = data

    return train.entrypoint(trainset)


def test_rmse_equals_benchmarks(model, data):
    _, testset = data
    
    rmse = evaluate.entrypoint(model, testset)

     # assert that the accuracy is 0.95 +/- 0.05
    assert rmse == pytest.approx(BENCHMARK_SVD_SCORE, rel=0.1)
    