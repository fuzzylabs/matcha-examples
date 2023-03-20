import pytest

import os
import logging

from zenml.logger import disable_logging
from zenml.post_execution import get_run

from pipelines import recommendation_pipeline

from steps import (
    load_data,
    train,
    evaluate,
)

from materializer import SurpriseMaterializer


from surprise import SVD


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPECTED_DATA_LENGTH = 100000
BENCHMARK_SVD_SCORE = 0.934


@pytest.fixture
def full_pipeline():
    return recommendation_pipeline


@pytest.fixture
def pipeline_configuration_path():
    return BASE_DIR + "/test_pipeline_config.yaml"


@pytest.fixture()
def pipeline_run(full_pipeline, pipeline_configuration_path):
    pipeline = full_pipeline(
        load_data().configure(output_materializers=SurpriseMaterializer),
        train().configure(output_materializers=SurpriseMaterializer),
        evaluate(),
    )

    with disable_logging(log_level=logging.INFO):
        pipeline.run(config_path=pipeline_configuration_path, unlisted=True)

    return get_run(name='test-pipeline')


def test_pipeline_executes(pipeline_run):
    rmse = pipeline_run.get_step(step="evaluate").output.read()

    assert rmse == pytest.approx(BENCHMARK_SVD_SCORE, rel=0.1)


def test_pipeline_loads_and_splits_correctly(pipeline_run, data_parameters):
    step_outputs = pipeline_run.get_step(step="load_data").outputs
    trainset = step_outputs["trainset"].read()
    testset = step_outputs["testset"].read()

    expected_size_train = int(EXPECTED_DATA_LENGTH * (1 - data_parameters.test_size))
    expected_size_test = int(EXPECTED_DATA_LENGTH * data_parameters.test_size)

    assert trainset.n_ratings + len(testset) == EXPECTED_DATA_LENGTH

    assert trainset.n_ratings == expected_size_train
    assert len(testset) == expected_size_test


def test_correct_model_type(pipeline_run):
    step_output = pipeline_run.get_step(step='train').output.read()

    assert isinstance(step_output, SVD)
