"""Test suite to test the convert_to_hg_dataset step."""
import pytest
import datasets

from steps.convert_to_hg_dataset_step import convert_to_hg_dataset


@pytest.fixture
def mock_data() -> dict:
    """Mock input dictionary data for step.

    Returns:
        dict: Dictionary with mocked input data
    """
    mock_data = {"data": {"original_text": "I am a text!",
                          "reference_summary": "I am a summary!"}}
    return mock_data


def test_convert_to_hg_dataset_step(mock_data: dict):
    """Test the convert_to_hg_dataset step.

    Args:
        mock_data (dict): Fixture to mock input data
    """
    hg_dataset = convert_to_hg_dataset.entrypoint(mock_data)

    expected_features = ["text", "summary"]

    # Check if the output is a huggingface `Dataset` object
    assert isinstance(hg_dataset, datasets.Dataset)

    # Check if the output has the expected number of features
    assert len(hg_dataset.features) == len(expected_features)

    # Check if the output has the expected features
    assert all([feat in hg_dataset.features for feat in expected_features])

    # Check if the all features are a `tf.Tensor` object
    assert all([isinstance(hg_dataset[feat][0], str) for feat in expected_features])
