"""Test suite to test the download data step."""
import pytest
from typing import Iterator
from types import SimpleNamespace
import tempfile
import os
from unittest import mock

from steps.download_data_step import download_dataset


@pytest.fixture
def temp_testing_directory() -> Iterator[str]:
    """A fixture for creating and removing temporary test directory for storing and moving files.

    Yields:
        str: a path to temporary directory for storing and moving files from tests.
    """
    temp_dir = tempfile.TemporaryDirectory()

    # tests are executed at this point
    yield temp_dir.name

    # delete temp folder
    temp_dir.cleanup()


@pytest.fixture
def get_params(temp_testing_directory: str) -> dict:
    """Mock parameters required for step.

    Args:
        temp_testing_directory (str): Path to temporary directory

    Returns:
        dict: Parameters for step.
    """
    params = SimpleNamespace()
    params.data_dir = temp_testing_directory
    return params


def test_download_data_step(get_params: dict):
    """Test the download data step.

    Args:
        get_params (dict): Fixture containing paramters for step.
    """
    dummy_dict = {'text': 'summary'}
    with mock.patch("requests.get") as mockresponse:
        mockresponse.return_value.status_code = 200
        mockresponse.return_value.json.return_value = dummy_dict

        data = download_dataset.entrypoint(get_params)

        # Check if returned data matches expected data
        assert data == dummy_dict

        # Check if folder is created
        assert os.path.exists(get_params.data_dir)

        # Check if file is created inside folder
        file_path = os.path.join(get_params.data_dir, "summarization_dataset.json")
        assert os.path.exists(file_path)


def test_download_data_step_invalid_url(get_params: dict):
    """Test the download data step when invalid url is passed.

    Args:
        get_params (dict): Fixture containing paramters for step.
    """
    dummy_dict = {'text': 'summary'}
    with mock.patch("requests.get") as mockresponse:
        mockresponse.return_value.status_code = 404
        mockresponse.return_value.json.return_value = dummy_dict

        with pytest.raises(Exception) as exc_info:
            _ = download_dataset.entrypoint(get_params)

    assert (
        str(exc_info.value)
        == "Error downloading dataset with response: 404"
    )
