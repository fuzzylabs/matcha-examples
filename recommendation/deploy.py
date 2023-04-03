"""Deploy the recommendation example model."""
from steps import (
    fetch_model,
    deploy_model
)
from pipelines import deploy_recommendation_pipeline
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
    pipeline = deploy_recommendation_pipeline(
        fetch_model=fetch_model().configure(output_materializers=SurpriseMaterializer),
        deploy_model=seldon_custom_model_deployer_step(
            config=SeldonDeployerStepParameters(
                service_config=SeldonDeploymentConfig(
                    model_name="seldon-surprise-custom-model",
                    replicas=1,
                    implementation="custom",
                    resources={"requests": {"cpu": "100m", "memory": "100m"}},
                ),
                timeout=240,
                custom_deploy_parameters=CustomDeployParameters(
                    predict_function="steps.deploy_model.custom_predict"
                ),
            )
        )
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