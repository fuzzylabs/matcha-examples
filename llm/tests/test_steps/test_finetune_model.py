"""Test finetune_model step."""
from unittest import mock

from steps.finetune_model import finetune_model, TuningParameters
import pytest
from datasets import Dataset
from transformers import PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel


@pytest.fixture
def params() -> TuningParameters:
    """Mock parameters required for step.

    Returns:
        GetHuggingfaceModelParameters: Parameters for step.
    """
    return TuningParameters(
        learning_rate=2e-5,
        weight_decay=0.01,
        use_cuda=False,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        epochs=5,
        load_best_model_at_end=True
    )

@pytest.fixture
def test_model() -> PreTrainedModel:
    """Get test dummy huggingface model.

    Returns:
        PreTrainedModel: test model
    """
    return PreTrainedModel(PretrainedConfig())

@pytest.fixture
def test_tokenizer() -> PreTrainedTokenizerBase:
    """Get test dummy huggingface tokenizer.

    Returns:
        PreTrainedTokenizerBase: test tokenizer
    """
    return PreTrainedTokenizerBase()

@pytest.fixture
def test_dataset() -> Dataset:
    """Get empty dataset.

    Returns:
        Dataset: empty dataset
    """
    return Dataset.from_dict({"train": [], "test": []})


@pytest.fixture
def expected_training_args() -> dict:
    """Get expected training arguments.

    Returns:
        dict: expected training arguments
    """
    return {
        "output_dir": "model",
        "learning_rate": 2e-5,
        "evaluation_strategy": 'epoch',
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "weight_decay": 0.01,
        "save_total_limit": 1,
        "save_strategy": 'epoch',
        "num_train_epochs": 5,
        "predict_with_generate": True,
        "no_cuda": True,
        "fp16": False,
        "load_best_model_at_end": True,
    }


def test_finetune_model(
    params: TuningParameters,
    test_tokenizer: PreTrainedTokenizerBase,
    test_model: PreTrainedModel,
    test_dataset: Dataset,
    expected_training_args: dict
):
    """Test finetune_model step fine-tunes a provided model.

    Args:
        params (TuningParameters): step parameters
        test_tokenizer (PreTrainedTokenizerBase): test tokenizer
        test_model (PreTrainedModel): test model
        test_dataset (Dataset): test empty dataset
        expected_training_args (dict): dictionary of expected training arguments
    """
    with mock.patch("steps.finetune_model.Seq2SeqTrainer") as mock_trainer, \
            mock.patch("steps.finetune_model.Seq2SeqTrainingArguments") as mock_trainer_args, \
            mock.patch("steps.finetune_model.DataCollatorForSeq2Seq") as mock_data_collator:
        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.tokenizer = test_tokenizer
        mock_trainer_instance.model = test_model

        tuned_tokenizer, tuned_model = finetune_model.entrypoint(params, test_tokenizer, test_model, test_dataset)

        mock_trainer_args.assert_called_with(**expected_training_args)
        mock_trainer.assert_called_with(
            model=test_model,
            args=mock_trainer_args.return_value,
            train_dataset=test_dataset["train"],
            eval_dataset=test_dataset["test"],
            tokenizer=test_tokenizer,
            data_collator=mock_data_collator.return_value
        )
        assert isinstance(tuned_tokenizer, PreTrainedTokenizerBase)
        assert isinstance(tuned_model, PreTrainedModel)

