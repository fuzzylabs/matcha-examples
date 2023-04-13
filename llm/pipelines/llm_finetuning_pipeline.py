"""Pipeline for the llm finetuning model."""
from zenml.pipelines import pipeline


@pipeline
def llm_finetuning_pipeline(download_dataset, convert_to_hg_dataset):
    """Pipeline for llm fine-tuning on summarization dataset.

    Args:
        download_dataset: A step to download the summarization dataset.
        convert_to_hg_dataset: A step to convert summarization dataset into
                            huggingface dataset format.
    """
    # Download the summarization dataset
    data = download_dataset()

    # Convert dataset into huggingface dataset format
    dataset = convert_to_hg_dataset(data)
