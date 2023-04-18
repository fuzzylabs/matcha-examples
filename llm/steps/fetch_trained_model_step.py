"""Fetch trained model for deploying."""
import os
from zenml.steps import step, BaseParameters, Output
from zenml.logger import get_logger

logger = get_logger(__name__)
REQUIRED_FILES = ["pytorch.bin", "tokenizer.json", "tokenizer_config.json"]


class FetchModelParams(BaseParameters):

    # Path to the folder containing trained checkpoints
    trained_model_path = "models/checkpoint-1500"


@step
def fetch_trained_model(params: FetchModelParams) -> Output(model_uri=str, decision=bool):
    """Fetch path to trained model and decision to deploy LLM model.

    Args:
        params (FetchModelParams): Parameters for fetching trained model.

    Returns:
        str: Path to folder containing trained checkpoints
        bool: Decision whether to deploy the model.
    """
    base_folder = os.path.dirname(params.trained_model_path)
    if not os.path.exists(base_folder):
        err_msg = f"Folder {base_folder} does not exist."
        logger.error(err_msg)
        raise ValueError(err_msg)

    for file in REQUIRED_FILES:
        file_path = os.path.join(params.trained_model_path, file)
        if not os.path.exists(file_path):
            err_msg = f"File {file_path} does not exist."
            logger.error(err_msg)
            raise ValueError(err_msg)

    return params.trained_model_path, True
