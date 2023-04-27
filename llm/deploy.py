"""Run the LLM deployment pipeline."""
from zenml.logger import get_logger

from steps.build_docker_image_step import build_docker_image
from steps.deploy_model_step import deploy_llm_model
from steps.fetch_trained_model_step import fetch_model
from pipelines.llm_deployment_pipeline import llm_deployment_pipeline

logger = get_logger(__name__)


def run_llm_deploy_pipeline():
    """Run all steps in llm deploy pipeline."""
    pipeline = llm_deployment_pipeline(fetch_model(), build_docker_image(), deploy_llm_model())
    pipeline.run(config_path="pipelines/config_llm_deployment_pipeline.yaml")


def main():
    """Run all pipelines."""
    logger.info("Running LLM deployment pipeline.")
    run_llm_deploy_pipeline()


if __name__ == "__main__":
    main()
