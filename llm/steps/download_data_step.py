"""Download summarization dataset for fine-tuning LLM."""
import os
import requests
import json

from zenml.logger import get_logger
from zenml.steps import step, BaseParameters


logger = get_logger(__name__)
DATASET_URL = "https://raw.githubusercontent.com/lauramanor/legal_summarization/master/all_v1.json"


class DownloadDataParams(BaseParameters):
    """Parameters for downloading dataset."""

    # Path to directory where dataset will be downloaded
    data_dir: str = "data"


@step
def download_dataset(params: DownloadDataParams) -> dict:
    """Zenml step to download summarization dataset.

    Args:
        params (DownloadDataParams): Parameters for downloading dataset.

    Returns:
        dict: Dataset in dictionary format.
    """
    # Check if dataset already exists
    data_path = os.path.join(params.data_dir, "summarization_dataset.json")
    if os.path.exists(data_path):
        logger.info(f"Dataset already exists at {data_path}")

        json_data = open(data_path).read()
        data = json.loads(json_data)
        return data

    else:
        logger.info("Downloading dataset")
        os.makedirs(params.data_dir, exist_ok=True)

        # Get the dataset from the url
        response = requests.get(DATASET_URL)
        data = response.json()

        # Write data to json file
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Dataset downloaded to {params.data_dir}")
        return data
