"""LLM finetuning step."""
from typing import Tuple

import torch
from zenml.steps import BaseParameters, step, Output
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from datasets import Dataset


class TuningParameters(BaseParameters):
    """Fine-tuning step parameters."""
    # Learning rate
    learning_rate: float

    # Weight decay
    weight_decay: float

    # Use CUDA for training
    use_cuda: bool

    # Batch size per device for training
    per_device_train_batch_size: int

    # Batch size per device for evaluation
    per_device_eval_batch_size: int

    # Number of epochs to run tuning for
    epochs: int

    # Load the best checkpoint at the end flag
    load_best_model_at_end: bool


def prepare_training_args(params: TuningParameters) -> Seq2SeqTrainingArguments:
    """Prepare training arguments.

    Args:
        params (TuningParameters): tuning step parameters

    Returns:
        Seq2SeqTrainingArguments: training arguments
    """
    use_cuda = torch.cuda.is_available() if params.use_cuda else False

    training_args = Seq2SeqTrainingArguments(
        output_dir="model",  # can be hardcoded to anything, since we are not using this directory afterwards
        learning_rate=params.learning_rate,
        evaluation_strategy="epoch",
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        weight_decay=params.weight_decay,
        save_total_limit=1,
        save_strategy="epoch",
        num_train_epochs=params.epochs,
        predict_with_generate=True,
        no_cuda=not use_cuda,
        fp16=use_cuda,
        load_best_model_at_end=params.load_best_model_at_end
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
        params: TuningParameters,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        data: Dataset
) -> Output(tokenizer=PreTrainedTokenizerBase, model=PreTrainedModel):
    """A step to fine-tune a pre-trained model.

    Args:
        params (TuningParameters): Tuning parameters
        tokenizer (PreTrainedTokenizerBase): A pre-trained tokenizer
        model (PreTrainedModel): A pre-trained model
        data (Dataset): A dataset to fine-tune with

    Returns:
        PreTrainedTokenizerBase: a tuned tokenizer
        PreTrainedModel: a tuned model
    """
    training_args = prepare_training_args(params)
    tuned_tokenizer, tuned_model = train(tokenizer, model, data, training_args)
    return tuned_tokenizer, tuned_model
