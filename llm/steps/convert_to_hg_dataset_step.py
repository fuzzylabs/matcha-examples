"""Convert a dataset to the huggingface dataset format."""
import pandas as pd

from zenml.logger import get_logger
from zenml.steps import step
from datasets import Dataset

logger = get_logger(__name__)


@step
def convert_to_hg_dataset(data: dict) -> Dataset:
    """Convert a dataset to the huggingface dataset format.

    Args:
        data (dict): dataset in `dict` format.

    Returns:
        Dataset: dataset in huggingface `Dataset` format.
    """
    # Extract text and summary from dataset
    texts, summaries = [], []
    for _, v in data.items():
        texts.append(v["original_text"])
        summaries.append(v["reference_summary"])

    df = pd.DataFrame({"text": texts, "summary": summaries})
    dataset = Dataset.from_pandas(df)

    logger.info("Total number of examples in dataset: {}".format(len(dataset)))
    return dataset
