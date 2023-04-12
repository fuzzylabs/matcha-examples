"""Functions required by ZenML and Seldon to deploy a custom SVD recommender model to Seldon Core."""
import numpy as np
from typing import Any, Dict, List, Union
from zenml.logger import get_logger

logger = get_logger(__name__)
Array_Like = Union[np.ndarray, List[Any], str, bytes, Dict[str, Any]]


def custom_predict(
    model: Any,
    request: Array_Like,
) -> Array_Like:
    """Custom Prediction function for SVD models.
    
    Request input is in the format: [{"iid": "2", "uid": "26"}, {"iid": "11", "uid": "7"}] 
    where each dictionary is a sample containing a user ID and a item ID to get an expected rating for.

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
        inputs.append(pred._asdict())
        
    return inputs
    