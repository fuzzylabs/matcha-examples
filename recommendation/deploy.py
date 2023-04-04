"""Deploy the recommendation example model."""
from steps.fetch_model import fetch_model
from steps.deployer import seldon_pytorch_custom_deployment
from steps.deployment_trigger import deployment_trigger
from pipelines.deploy_recommendation_pipeline import recommendation_deployment_pipeline
from materializer import SurpriseMaterializer

from zenml.logger import get_logger
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

from zenml.integrations.seldon.steps import (
  seldon_custom_model_deployer_step, 
    SeldonDeployerStepParameters,
    CustomDeployParameters,
)
from zenml.integrations.seldon.services import SeldonDeploymentConfig



logger = get_logger(__name__)


def run_deployment_pipeline():
    """Run all steps in the deployment pipeline."""
    pipeline = recommendation_deployment_pipeline(
        fetch_model().configure(output_materializers=SurpriseMaterializer),
        deployment_trigger(),
        deploy_model=seldon_pytorch_custom_deployment,
    )
    pipeline.run(config_path="pipelines/config_deploy_recommendation_pipeline.yaml")
    logger.info(
        f"Visit: {get_tracking_uri()}\n "
        "To inspect your experiment runs within the mlflow UI.\n"
    )


def main():
    """Run pipeline."""
    logger.info("Running deployment pipeline")
    run_deployment_pipeline()


if __name__ == "__main__":
    main()