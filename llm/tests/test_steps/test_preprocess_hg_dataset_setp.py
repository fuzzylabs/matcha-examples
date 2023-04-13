"""Test suite to test the preprocess_hg_dataset step."""
import pytest
from types import SimpleNamespace

from datasets import Dataset
from transformers import BatchEncoding

from steps.convert_to_hg_dataset_step import convert_to_hg_dataset
from steps.preprocess_hg_dataset_step import preprocess_dataset, preprocess_function


@pytest.fixture
def get_params() -> dict:
    """Mock parameters required for step.

    Returns:
        dict: Parameters for step.
    """
    params = SimpleNamespace()
    params.model_name = "google/flan-t5-base"
    params.prefix = "summarize: "
    params.input_max_length = 4096
    params.target_max_length = 512
    params.test_size = 0.2
    return params


@pytest.fixture
def mock_data() -> dict:
    """Mock input dictionary data for step.

    Returns:
        dict: Dictionary with mocked input data
    """
    mock_data = {"data": {"original_text": "I am a text!",
                          "reference_summary": "I am a summary!"}}
    return mock_data


@pytest.fixture
def mock_hf_dataset(mock_data: dict) -> Dataset:
    hg_dataset = convert_to_hg_dataset.entrypoint(mock_data)
    return hg_dataset


def test_preprocess_function(mock_hf_dataset: Dataset, get_params: dict):
    """Test the preprocess_function function."""
    tokenized_dataset = preprocess_function(mock_hf_dataset,
                                            get_params.model_name,
                                            get_params.prefix,
                                            get_params.input_max_length,
                                            get_params.target_max_length)

    expected_features = ['input_ids', 'attention_mask', 'labels']

    expected_labels = [[27, 183, 3, 9, 9251, 55, 1]]
    expected_input_ids = [[21603, 10, 27, 183, 3, 9, 1499, 55, 1]]

    # Check if the output is a huggingface `Dataset` object
    assert isinstance(tokenized_dataset, BatchEncoding)

    # Check if the output contains the expected features
    assert all([feat in tokenized_dataset.keys() for feat in expected_features])

    # Check if length of each feature is correct
    assert all([len(v) == len(mock_hf_dataset) for _, v in tokenized_dataset.items()])

    # Check if input and labels are correct
    assert tokenized_dataset['input_ids'] == expected_input_ids
    assert tokenized_dataset['labels'] == expected_labels
