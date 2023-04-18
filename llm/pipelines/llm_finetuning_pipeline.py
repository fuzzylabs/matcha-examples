"""Pipeline for the llm finetuning model."""
from zenml.pipelines import pipeline


@pipeline
def llm_finetuning_pipeline(
        download_dataset,
        convert_to_hg_dataset,
        get_huggingface_model,
        preprocess_dataset,
        finetune_model,
):
    """Pipeline for llm fine-tuning on summarization dataset.

    Args:
        download_dataset: A step to download the summarization dataset.
        convert_to_hg_dataset: A step to convert summarization dataset into
                            huggingface dataset format.
        get_huggingface_model: Get pre-trained model from huggingface
        preprocess_dataset: A step to preprocess, tokenize and split the summarization dataset.
    """
    # Download the summarization dataset
    data = download_dataset()

    # Convert dataset into huggingface dataset format
    dataset = convert_to_hg_dataset(data)

    # Get the pre-trained model
    tokenizer, model = get_huggingface_model()

    # Preprocess, tokenize and split dataset
    tokenized_data = preprocess_dataset(dataset, tokenizer)

    # Fine-tune
    tuned_tokenizer, tuned_model = finetune_model(tokenizer, model, tokenized_data)

