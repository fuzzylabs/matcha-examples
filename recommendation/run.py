"""Run the recommendation example pipeline."""
import click
from steps.load_data_step import load_data
from steps.train_step import train
from steps.evaluate_step import evaluate
from pipelines.recommendation_pipeline import recommendation_pipeline
from steps.fetch_model import fetch_model
from steps.deployer import seldon_surprise_custom_deployment
from steps.deployment_trigger import deployment_trigger
from pipelines.deploy_recommendation_pipeline import recommendation_deployment_pipeline
from materializer import SurpriseMaterializer

from zenml.logger import get_logger
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


logger = get_logger(__name__)


def run_recommendation_pipeline():
    """Run all steps in the example pipeline."""
    pipeline = recommendation_pipeline(
        load_data().configure(output_materializers=SurpriseMaterializer),
        train().configure(output_materializers=SurpriseMaterializer),
        evaluate(),
    )
    pipeline.run(config_path="pipelines/config_recommendation_pipeline.yaml")
    
    logger.info(
        f"Visit: {get_tracking_uri()}\n "
        "To inspect your experiment runs within the mlflow UI.\n"
    )
    
    
def run_deployment_pipeline():
    """Run all steps in deployment pipeline."""
    deploy_pipeline = recommendation_deployment_pipeline(
        fetch_model().configure(output_materializers=SurpriseMaterializer),
        deployment_trigger(),
        deploy_model=seldon_surprise_custom_deployment,
    )
    deploy_pipeline.run(config_path="pipelines/config_deploy_recommendation_pipeline.yaml")
    
    
@click.command()
@click.option("--train", "-t", is_flag=True, help="Run training pipeline")
def main(train: bool):
    """Run all pipelines.
    
    args:
        train (bool): Flag for running the training pipeline.
    """
    if train:
        logger.info("Running recommendation training pipeline.")
        run_recommendation_pipeline()
    
    if not train:
        logger.info("Running recommendation training pipeline.")
        run_recommendation_pipeline()
        
        logger.info("Running deployment pipeline.")
        run_deployment_pipeline()


if __name__ == "__main__":
    main()