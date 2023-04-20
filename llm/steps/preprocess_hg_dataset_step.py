"""Preprocess, tokenize and split the huggingface dataset into train/test set."""
from functools import partial

from zenml.logger import get_logger
from zenml.steps import step, BaseParameters
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, BatchEncoding
from transformers import PreTrainedTokenizerBase

logger = get_logger(__name__)


class PreprocessParameters(BaseParameters):
    """Parameters for preprocessing the Huggingface dataset."""

    # Prefix to be added to the input (required for T5 LLM family)
    prefix: str = "summarize: "

    # Max length of the input text
    input_max_length: int = 4096

    # Max length of the target summary
    target_max_length: int = 512

    # Split ratio for train/test
    test_size: float = 0.2


def preprocess_function(dataset: Dataset,
                        tokenizer: PreTrainedTokenizerBase,
                        prefix: str,
                        input_max_length: int,
                        target_max_length: int) -> BatchEncoding:
    """Preprocess and tokenize the huggingface dataset.

    Args:
        dataset (Dataset): Dataset to preprocess and tokenize.
        tokenizer (str): Huggingface tokenizer.
        prefix (str): Prefix to add so that T5 model knows this is a summarization task.
        input_max_length (int): Max length of the input text. Truncate sequences to be no longer than this length.
        target_max_length (int): Max length of the target summary. Truncate sequences to be no longer than this length.

    Returns:
        BatchEncoding: Tokenized input and targets.
    """
    # Preprocess input by adding the prefix so T5 knows this is a summarization task.
    inputs = [prefix + doc for doc in dataset["text"]]

    # Tokenize input and target
    model_inputs = tokenizer(inputs,
                             max_length=input_max_length,
                             truncation=True)
    labels = tokenizer(text_target=dataset["summary"],
                       max_length=target_max_length,
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


@step
def preprocess_dataset(dataset: Dataset, tokenizer: PreTrainedTokenizerBase, params: PreprocessParameters) -> DatasetDict:
    """Preprocess the huggingface dataset.

    Args:
        dataset (Dataset): Dataset to preprocess, tokenize and split.
        tokenizer (str): Huggingface tokenizer.
        params (PreprocessParameters): Parameters for preprocessing the dataset.

    Returns:
        DatasetDict: Tokenized dataset split into train and test.
    """
    # Tokenize and preprocess dataset
    tokenized_data = dataset.map(partial(preprocess_function,
                                         tokenizer=tokenizer,
                                         prefix=params.prefix,
                                         input_max_length=params.input_max_length,
                                         target_max_length=params.target_max_length
                                         ),
                                 batched=True)

    # Split into train and test
    tokenized_data = tokenized_data.train_test_split(test_size=params.test_size)

    logger.info(f"Number of examples in training set: {len(tokenized_data['train'])}")
    logger.info(f"Number of examples in test set: {len(tokenized_data['test'])}")
    return tokenized_data
