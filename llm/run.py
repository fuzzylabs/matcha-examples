"""Run the LLM finetuning pipeline."""
from zenml.logger import get_logger

from steps.download_data_step import download_dataset
from pipelines.llm_finetuning_pipeline import llm_finetuning_pipeline

logger = get_logger(__name__)


def run_llm_finetuning_pipeline():
    """Run all steps in the llm finetuning pipeline."""
    pipeline = llm_finetuning_pipeline(download_dataset())
    pipeline.run(config_path="pipelines/config_llm_finetuning_pipeline.yaml")


def main():
    """Run all pipelines."""
    logger.info("Running LLM fine-tuning pipeline.")
    run_llm_finetuning_pipeline()


if __name__ == "__main__":
    main()
