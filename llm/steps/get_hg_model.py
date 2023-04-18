"""Get Huggingface model."""
from zenml.steps import step, Output, BaseParameters
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, PreTrainedModel


class GetHuggingfaceModelParameters(BaseParameters):
    """Parameters for downloading the huggingface model."""

    # Huggingface model name
    model_name: str


@step
def get_huggingface_model(params: GetHuggingfaceModelParameters) -> Output(
    tokenizer=PreTrainedTokenizerBase,
    model=PreTrainedModel
):
    tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(params.model_name)

    return tokenizer, model
