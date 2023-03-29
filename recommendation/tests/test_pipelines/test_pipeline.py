"""Integration tests for the pipeline as a whole."""
import pytest

import os
import logging

from zenml.logger import disable_logging
from zenml.post_execution import get_unlisted_runs
from zenml.post_execution.pipeline_run import PipelineRunView

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
BENCHMARK_SVD_SCORE = 0.93


@pytest.fixture(scope="class", autouse=True)
def pipeline_run():
    """Set up fixture for running the pipeline."""
    pipeline = recommendation_pipeline(
        load_data().configure(output_materializers=SurpriseMaterializer),
        train().configure(output_materializers=SurpriseMaterializer),
        evaluate(),
    )

    with disable_logging(log_level=logging.INFO):
        pipeline.run(config_path=BASE_DIR + "/test_pipeline_config.yaml", unlisted=True)


@pytest.fixture()
def get_pipeline_run() -> PipelineRunView:
    """Get the most recent pipeline run.

    Args:
        pipeline_run: the pipeline run fixture which is executed once.

    Returns:
        PipelineRunView: the test run
    """
    return get_unlisted_runs()[0]


def test_pipeline_executes(get_pipeline_run: PipelineRunView):
    """Test the model training result from the the pipeline run.

    Args:
        get_pipeline_run (PipelineRunView): pipeline run.
    """
    rmse = get_pipeline_run.get_step(step="evaluate").output.read()

    assert rmse == pytest.approx(BENCHMARK_SVD_SCORE, rel=0.1)


def test_pipeline_loads_and_splits_correctly(get_pipeline_run: PipelineRunView, data_parameters: dict):
    """Test whether the pipeline splits data into train and test correctly.

    Args:
        get_pipeline_run (PipelineRunView): the pipeline run
        data_parameters (dict): parameters for train test split
    """
    step_outputs = get_pipeline_run.get_step(step="load_data").outputs
    trainset = step_outputs["trainset"].read()
    testset = step_outputs["testset"].read()

    expected_size_train = int(EXPECTED_DATA_LENGTH * (1 - data_parameters.test_size))
    expected_size_test = int(EXPECTED_DATA_LENGTH * data_parameters.test_size)

    assert trainset.n_ratings + len(testset) == EXPECTED_DATA_LENGTH

    assert trainset.n_ratings == expected_size_train
    assert len(testset) == expected_size_test


def test_correct_model_type(get_pipeline_run: PipelineRunView):
    """Test whether the pipeline return the correct model type.

    Args:
        get_pipeline_run (PipelineRunView): the pipeline run
    """
    step_output = get_pipeline_run.get_step(step='train').output.read()

    assert isinstance(step_output, SVD)
