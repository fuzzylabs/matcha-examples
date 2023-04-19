"""Run the LLM finetuning pipeline."""
from zenml.logger import get_logger

from steps.download_data_step import download_dataset
from steps.convert_to_hg_dataset_step import convert_to_hg_dataset
from steps.fetch_trained_model_step import fetch_model
from steps.preprocess_hg_dataset_step import preprocess_dataset
from pipelines.llm_finetuning_pipeline import llm_finetuning_pipeline
from pipelines.llm_deployment_pipeline import llm_deployment_pipeline

logger = get_logger(__name__)


def run_llm_finetuning_pipeline():
    """Run all steps in the llm finetuning pipeline."""
    pipeline = llm_finetuning_pipeline(download_dataset(),
                                       convert_to_hg_dataset(),
                                       preprocess_dataset())
    pipeline.run(config_path="pipelines/config_llm_finetuning_pipeline.yaml")


def run_llm_deploy_pipeline():
    """Run all steps in llm deploy pipeline."""
    pipeline = llm_deployment_pipeline(fetch_model())
    pipeline.run(config_path="pipelines/config_llm_deployment_pipeline.yaml")


def main():
    """Run all pipelines."""
    logger.info("Running LLM fine-tuning pipeline.")
    run_llm_finetuning_pipeline()
    run_llm_deploy_pipeline()


if __name__ == "__main__":
    main()
