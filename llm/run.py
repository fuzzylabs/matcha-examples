"""Run the LLM finetuning pipeline."""
from zenml.logger import get_logger

from steps.finetune import finetune_model
from steps.get_hg_model import get_huggingface_model
from steps.download_data_step import download_dataset
from steps.convert_to_hg_dataset_step import convert_to_hg_dataset
from steps.preprocess_hg_dataset_step import preprocess_dataset
from pipelines.llm_finetuning_pipeline import llm_finetuning_pipeline

logger = get_logger(__name__)


def run_llm_finetuning_pipeline():
    """Run all steps in the llm finetuning pipeline."""
    pipeline = llm_finetuning_pipeline(
        download_dataset(),
        convert_to_hg_dataset(),
        get_huggingface_model(),
        preprocess_dataset(),
        finetune_model()
    )
    pipeline.run(config_path="pipelines/config_llm_finetuning_pipeline.yaml")


def main():
    """Run all pipelines."""
    logger.info("Running LLM fine-tuning pipeline.")
    run_llm_finetuning_pipeline()


if __name__ == "__main__":
    main()
