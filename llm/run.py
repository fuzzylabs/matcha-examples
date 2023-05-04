"""Run the LLM finetuning and deployment pipeline."""
import click
from zenml.logger import get_logger

from steps.finetune_model import finetune_model
from steps.get_hg_model import get_huggingface_model
from steps.download_data_step import download_dataset
from steps.convert_to_hg_dataset_step import convert_to_hg_dataset
from steps.preprocess_hg_dataset_step import preprocess_dataset
from steps.deploy_model_step import seldon_llm_custom_deployment
from steps.fetch_trained_model_step import fetch_model

from pipelines.llm_deployment_pipeline import llm_deployment_pipeline
from pipelines.llm_pipeline import llm_pipeline

logger = get_logger(__name__)


def run_llm_pipeline():
    """Run all steps in the llm finetuning pipeline."""
    pipeline = llm_pipeline(
        download_dataset(),
        convert_to_hg_dataset(),
        get_huggingface_model(),
        preprocess_dataset(),
        finetune_model(),
    )
    pipeline.run(config_path="pipelines/config_llm_pipeline.yaml")


def run_llm_deploy_pipeline():
    """Run all steps in llm deploy pipeline."""
    pipeline = llm_deployment_pipeline(fetch_model(), seldon_llm_custom_deployment)
    pipeline.run(config_path="pipelines/config_llm_deployment_pipeline.yaml")


@click.command()
@click.option("--train", "-t", is_flag=True, help="Run training pipeline")
@click.option("--deploy", "-d", is_flag=True, help="Run the deployment pipeline")
def main(train: bool, deploy: bool):
    """Run all pipelines.

    args:
        train (bool): Flag for running the training pipeline.
        deploy (bool): Flag for running the deployment pipeline.
    """
    if train:
        """Run all pipelines."""
        logger.info("Running LLM fine-tuning pipeline.")
        run_llm_pipeline()

    if deploy:
        logger.info("Running LLM deployment pipeline.")
        run_llm_deploy_pipeline()


if __name__ == "__main__":
    main()
