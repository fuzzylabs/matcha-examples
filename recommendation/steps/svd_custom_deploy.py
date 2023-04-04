"""ZenML step to deploy a SVD recommender model to Seldon Core."""
import os
from typing import Any, Dict, List, Union

from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.logger import get_logger
from zenml.materializers import UnmaterializedArtifact
from zenml.steps import BaseParameters, StepContext, step

import numpy as np

logger = get_logger(__name__)

Array_Like = Union[np.ndarray, List[Any], str, bytes, Dict[str, Any]]


def pre_process(input: np.ndarray) -> np.ndarray:
    """Pre process the data to be used for prediction."""
    pass


def post_process(prediction: np.ndarray) -> str:
    """Pre process the data"""
    pass


def custom_predict(
    model: Any,
    request: Array_Like,
) -> Array_Like:
    """Custom Prediction function.

    The custom predict function is the core of the custom deployment, the 
    function is called by the custom deployment class defined for the serving 
    tool. The current implementation requires the function to get the model 
    loaded in the memory and a request with the data to predict.

    Args:
        model (Any): The model to use for prediction.
        request: The prediction response of the model is an array-like format.
    Returns:
        The prediction in an array-like format. (e.g: np.ndarray, 
        List[Any], str, bytes, Dict[str, Any])
    """
    inputs = []
    for instance in request:
        pred = model.predict(instance['uid'], instance['iid'])
        
        inputs.append(pred)
        
    return inputs
    