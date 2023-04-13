"""Pipeline for the llm finetuning model."""
from zenml.pipelines import pipeline


@pipeline
def llm_finetuning_pipeline(download_dataset):
    """Pipeline for llm fine-tuning on summarization dataset.

    Args:
        download_dataset: A step to download the summarization dataset. 
    """

    download_dataset()