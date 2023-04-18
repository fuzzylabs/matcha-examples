"""LLM finetuning step."""
from typing import Tuple

from zenml.steps import BaseParameters, step, Output
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from datasets import Dataset


class FinetuneParameters(BaseParameters):
    """Fine-tuning step parameters."""
    # Enable Transformers fp16 mixed precision
    fp16: bool


def prepare_training_args(fp16: bool) -> Seq2SeqTrainingArguments:
    """Prepare training arguments.

    Args:
        fp16 (bool): Enable Transformers fp16 mixed precision

    Returns:
        Seq2SeqTrainingArguments: training arguments
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=fp16,
        push_to_hub=False,
    )
    return training_args


def train(
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        data: Dataset,
        training_args: Seq2SeqTrainingArguments
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Perform sequence to sequence training.

    Args:
        tokenizer (PreTrainedTokenizerBase): Huggingface tokenizer to be used in fine-tuning.
        model (PreTrainedModel): Huggingface pre-trained model to fine-tune
        data (Dataset): dataset to fine-tune with
        training_args (Seq2SeqTrainingArguments): training arguments

    Returns:
        Tuple[PreTrainedTokenizerBase, PreTrainedModel]: fine-tuned tokenizer and model

    Raises:
        Exception: when tokenizer or model is missing from the trainer
    """
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.name_or_path)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    if trainer.tokenizer is None:
        raise Exception("Trainer's tokenizer is None")
    if trainer.model is None:
        raise Exception("Trainer's model is None")

    return trainer.tokenizer, trainer.model


@step
def finetune_model(
        params: FinetuneParameters,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        data: Dataset
) -> Output(tokenizer=PreTrainedTokenizerBase, model=PreTrainedModel):
    """A step to fine-tune a pre-trained model.

    Args:
        params (FinetuneParameters): Fine-tuning parameters
        tokenizer (PreTrainedTokenizerBase): A pre-trained tokenizer
        model (PreTrainedModel): A pre-trained model
        data (Dataset): A dataset to fine-tune with

    Returns:
        PreTrainedTokenizerBase: a fine-tuned tokenizer
        PreTrainedModel: a fine-tuned model
    """
    training_args = prepare_training_args(params.fp16)
    tuned_tokenizer, tuned_model = train(tokenizer, model, data, training_args)
    return tuned_tokenizer, tuned_model
