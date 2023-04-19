"""Tests for get_huggingface_model step."""
from unittest import mock

from transformers import PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel
from steps.get_hg_model import get_huggingface_model, GetHuggingfaceModelParameters
import pytest


@pytest.fixture
def params() -> GetHuggingfaceModelParameters:
    """Mock parameters required for step.

    Returns:
        GetHuggingfaceModelParameters: Parameters for step.
    """
    return GetHuggingfaceModelParameters(
        model_name="test_model"
    )


def test_get_huggingface_model(params: GetHuggingfaceModelParameters):
    """Test get_huggingface_model gets pre-trained tokenizer and model

    Args:
        params (GetHuggingfaceModelParameters): test parameters
    """
    with (
        mock.patch("steps.get_hg_model.AutoTokenizer") as mock_tokenizer,
        mock.patch("steps.get_hg_model.AutoModelForSeq2SeqLM") as mock_model,
    ):
        mock_tokenizer.from_pretrained.return_value = PreTrainedTokenizerBase()
        mock_model.from_pretrained.return_value = PreTrainedModel(PretrainedConfig())
        tokenizer, model = get_huggingface_model.entrypoint(params)
        assert isinstance(tokenizer, PreTrainedTokenizerBase)
        assert isinstance(model, PreTrainedModel)
