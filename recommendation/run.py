"""Run the recommendation example pipeline."""
from steps import (
    load_data,
    train,
    evaluate,
)
from pipelines import recommendation_pipeline
from materializer import SurpriseMaterializer

from zenml.logger import get_logger


logger = get_logger(__name__)


def run_recommendation_pipeline():
    """Run all steps in the example pipeline."""
    pipeline = recommendation_pipeline(
        load_data().configure(output_materializers=SurpriseMaterializer),
        train().configure(output_materializers=SurpriseMaterializer),
        evaluate(),
    )
    pipeline.run(config_path="pipeline_config.yaml")


def main():
    """Run all pipeline."""
    logger.info("Running recommendation pipeline")
    run_recommendation_pipeline()


if __name__ == "__main__":
    main()