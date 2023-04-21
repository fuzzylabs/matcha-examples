"""Get Huggingface model."""
from zenml.steps import step, Output, BaseParameters
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, PreTrainedModel
from zenml.logger import get_logger

logger = get_logger(__name__)


class GetHuggingfaceModelParameters(BaseParameters):
    """Parameters for downloading the huggingface model."""

    # Huggingface model name
    model_name: str


@step
def get_huggingface_model(params: GetHuggingfaceModelParameters) -> Output(
    tokenizer=PreTrainedTokenizerBase,
    model=PreTrainedModel
):
    """A step to get Huggingface model from the hub.

    Args:
        params: step parameters

    Returns:
        PreTrainedTokenizerBase: a pre-trained tokenizer
        PreTrainedModel: a pre-trained model
    """
    logger.info(f"Loading model and tokenizer from HuggingFace hub with model_name = {params.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(params.model_name)

    return tokenizer, model
