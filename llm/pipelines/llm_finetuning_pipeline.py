"""Pipeline for the llm finetuning model."""
from zenml.pipelines import pipeline


@pipeline
def llm_finetuning_pipeline(download_dataset, convert_to_hg_dataset, preprocess_dataset):
    """Pipeline for llm fine-tuning on summarization dataset.

    Args:
        download_dataset: A step to download the summarization dataset.
        convert_to_hg_dataset: A step to convert summarization dataset into
                            huggingface dataset format.
        preprocess_dataset: A step to preprocess, tokenize and split the summarization dataset.
    """
    # Download the summarization dataset
    data = download_dataset()

    # Convert dataset into huggingface dataset format
    dataset = convert_to_hg_dataset(data)

    # Preprocess, tokenize and split dataset
    tokenized_data = preprocess_dataset(dataset)
