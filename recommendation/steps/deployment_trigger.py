"""A step to decide whether to deploy the model or not."""
from zenml.logger import get_logger
from zenml.post_execution import PipelineView, get_pipeline
from zenml.post_execution.artifact import ArtifactView
from zenml.steps import BaseParameters, Output, step

from surprise import SVD

logger = get_logger(__name__)


@step()
def deployment_trigger(
    model: SVD,
) -> Output(decision=bool):
    """Step to decide whether to deploy the model or not.

    Args:
        model (SVD): Model to check deployment.

    Returns:
        bool: Decision to deploy the model.
    """
    return True