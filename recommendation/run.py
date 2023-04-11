"""Run the recommendation example pipeline."""
from steps.load_data_step import load_data
from steps.train_step import train
from steps.evaluate_step import evaluate
from pipelines.recommendation_pipeline import recommendation_pipeline
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


def main():
    """Run all pipeline."""
    logger.info("Running recommendation pipeline")
    run_recommendation_pipeline()


if __name__ == "__main__":
    main()