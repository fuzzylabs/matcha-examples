"""Fetch trained model for deploying."""
from typing import Optional

from zenml.post_execution import PipelineView, get_pipeline
from zenml.post_execution.artifact import ArtifactView
from zenml.models.artifact_models import ArtifactResponseModel
from zenml.steps import step, BaseParameters, Output
from zenml.logger import get_logger

logger = get_logger(__name__)


class FetchModelParameters(BaseParameters):
    """Parameters for fetch model step."""

    # Name of the pipeline to fetch from
    pipeline_name: str

    # Step name to fetch artifact from.
    step_name: str

    # Optional pipeline version
    pipeline_version: Optional[int] = None


def get_output_from_step(pipeline: PipelineView, step_name: str) -> ArtifactView:
    """Fetch output from a step with last completed run in a pipeline.

    Args:
        pipeline (PipelineView): Post-execution pipeline class object.
        step_name (str): Name of step to fetch output from

    Returns:
        ArtifactView: Artifact data

    Raises:
        KeyError: If no step found with given name
        ValueError: If no output found for step
    """
    # Get the step with the given name from the last run
    try:
        # Get last completed run of the pipeline
        fetch_last_completed_run = pipeline.get_run_for_completed_step(step_name)

        logger.info(f"Run used: {fetch_last_completed_run}")

        # Get the output of the step from last completed run
        fetch_step = fetch_last_completed_run.get_step(step_name)

        logger.info(f"Step used: {fetch_step}")
    except KeyError as e:
        logger.error(f"No step found with name '{step_name}': {e}")
        raise e

    # Get the model artifacts from the step
    outputs = fetch_step.outputs
    if outputs is None:
        logger.error(f"No output found for step '{step_name}'")
        raise ValueError(f"No output found for step '{step_name}'")

    return outputs


@step
def fetch_model(
    params: FetchModelParameters,
) -> Output(model=ArtifactResponseModel, tokenizer=ArtifactResponseModel, decision=bool):
    """Step to fetch model artifacts from last run of nft_embedding pipeline.

    Args:
        params (FetchModelParameters): Parameters for fetch model step.

    Returns:
        ArtifactResponseModel: Artifact data for model.
        ArtifactResponseModel: Artifact data for tokenizer.
        bool: Decision to deploy the model.

    Raises:
        ValueError: if the pipeline does not exist
    """
    # Fetch pipeline by name
    pipeline: PipelineView = get_pipeline(
        params.pipeline_name, version=params.pipeline_version
    )

    if pipeline is None:
        logger.error(f"Pipeline '{params.pipeline_name}' does not exist")
        raise ValueError(f"Pipeline '{params.pipeline_name}' does not exist")

    logger.info(f"Pipeline: {pipeline}")

    # Fetch the output for step from the pipeline
    outputs = get_output_from_step(pipeline, params.step_name)

    model, tokenizer = outputs["model"], outputs["tokenizer"]
    return model, tokenizer, True
