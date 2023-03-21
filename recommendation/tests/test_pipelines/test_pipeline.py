"""Integration tests for the pipeline as a whole."""
import pytest

import os
import logging

import zenml
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
def full_pipeline() -> zenml.pipelines.base_pipeline.BasePipelineMeta:
    """Pytest fixture for running the all the steps in the zenml pipeline.

    Returns:
        zenml.pipelines.base_pipeline.BasePipelineMeta: ZenML pipeline
    """
    return recommendation_pipeline


@pytest.fixture
def pipeline_configuration_path() -> str:
    """A fixture to define the path to the configuration file.

    Returns:
        str: the path to the configuration file.
    """
    return BASE_DIR + "/test_pipeline_config.yaml"


@pytest.fixture()
def pipeline_run(full_pipeline: zenml.pipelines.base_pipeline.BasePipelineMeta, pipeline_configuration_path: str) -> zenml.post_execution.pipeline_run.PipelineRunView:
    """_summary_

    Args:
        full_pipeline (zenml.pipelines.base_pipeline.BasePipelineMeta): the recommendation pipeline
        pipeline_configuration_path (str): the test pipeline config path

    Returns:
        zenml.post_execution.pipeline_run.PipelineRunView: the pipeline run
    """
    pipeline = full_pipeline(
        load_data().configure(output_materializers=SurpriseMaterializer),
        train().configure(output_materializers=SurpriseMaterializer),
        evaluate(),
    )

    with disable_logging(log_level=logging.INFO):
        pipeline.run(config_path=pipeline_configuration_path, unlisted=True)

    return get_run(name='test-pipeline')


# def test_pipeline_executes(pipeline_run: zenml.post_execution.pipeline_run.PipelineRunView):
#     """Test the model training result from the the pipeline run.

#     Args:
#         pipeline_run (zenml.post_execution.pipeline_run.PipelineRunView): pipeline run.
#     """
#     rmse = pipeline_run.get_step(step="evaluate").output.read()

#     assert rmse == pytest.approx(BENCHMARK_SVD_SCORE, rel=0.1)


def test_pipeline_loads_and_splits_correctly(pipeline_run: zenml.post_execution.pipeline_run.PipelineRunView, data_parameters: dict):
    """Test whether the pipeline splits data into train and test correctly.

    Args:
        pipeline_run (zenml.post_execution.pipeline_run.PipelineRunView): the pipeline run
        data_parameters (dict): parameters for train test split
    """
    step_outputs = pipeline_run.get_step(step="load_data").outputs
    trainset = step_outputs["trainset"].read()
    testset = step_outputs["testset"].read()

    expected_size_train = int(EXPECTED_DATA_LENGTH * (1 - data_parameters.test_size))
    expected_size_test = int(EXPECTED_DATA_LENGTH * data_parameters.test_size)

    assert trainset.n_ratings + len(testset) == EXPECTED_DATA_LENGTH

    assert trainset.n_ratings == expected_size_train
    assert len(testset) == expected_size_test


# def test_correct_model_type(pipeline_run: zenml.post_execution.pipeline_run.PipelineRunView):
#     """Test whether the pipeline return the correct model type.

#     Args:
#         pipeline_run (zenml.post_execution.pipeline_run.PipelineRunView): the pipeline run
#     """
#     step_output = pipeline_run.get_step(step='train').output.read()

#     assert isinstance(step_output, SVD)
