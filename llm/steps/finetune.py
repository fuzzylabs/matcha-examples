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
    fp16: bool


def prepare_training_args(fp16: bool):
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model.name_or_path)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    smth = trainer.train()

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
    training_args = prepare_training_args(params.fp16)
    tuned_tokenizer, tuned_model = train(tokenizer, model, data, training_args)
    return tuned_tokenizer, tuned_model
