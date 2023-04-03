"""A step to fetch model artifacts for recommendation deployment pipeline."""
from typing import Optional, Tuple

from zenml.logger import get_logger
from zenml.post_execution import PipelineView, get_pipeline
from zenml.post_execution.artifact import ArtifactView
from zenml.steps import BaseParameters, Output, step

from surprise import SVD

logger = get_logger(__name__)

class FetchModelParameters(BaseParameters):
    """Parameters for fetch model step."""

    # Name of the pipeline to fetch from
    pipeline_name: str

    # Step name to fetch model from.
    step_name: str

    # Optional pipeline version
    pipeline_version: Optional[int] = None


def get_model_from_step(pipeline: PipelineView, fetch_model_step_name: str) -> SVD:
    """Fetch recommender model from specified pipeline and step name.

    Args:
        pipeline (PipelineView): Post-execution pipeline class object.
        fetch_model_step_name (str): Name of step to fetch model from.

    Returns:
        SVD: surprise SVD recommender model.
    """
    # Get the output from the step
    output = get_output_from_step(pipeline, fetch_model_step_name)

    # Read the model artifact from the output
    model = output.read()

    return model


@step
def fetch_model(
    params: FetchModelParameters,
) -> Output(decision=bool, model=SVD):
    """Step to fetch model artifacts from last run of the recommendation pipeline.

    Args:
        params (FetchModelParameters): Parameters for fetch model step.

    Returns:
        bool: Decision to deploy the model.
        SVD: Model artifacts.

    Raises:
        ValueError: if the pipeline does not exist
        KeyError: if step name parameter is not correct.
    """
    # Fetch pipeline by name
    pipeline: PipelineView | None = get_pipeline(
        params.pipeline_name, version=params.pipeline_version
    )

    if pipeline is None:
        logger.error(f"Pipeline '{params.pipeline_name}' does not exist")
        raise ValueError(f"Pipeline '{params.pipeline_name}' does not exist")

    logger.info(f"Pipeline: {pipeline}")

    # Fetch the model artifacts from the pipeline
    model = get_model_from_step(pipeline, params.step_name)

    return model