"""A pipeline to load, train and evaluate a simple recommendation model."""
from zenml.pipelines import pipeline
from zenml.config import DockerSettings

docker_settings = DockerSettings(apt_packages=["gcc", "build-essential"])


@pipeline(settings={"docker": docker_settings})
def recommendation_pipeline(
    load_data,
    train,
    evaluate,
):
    """Recommendation example pipeline.

    Steps
    1. load_data: This step load a built in dataset from the Surprise library and splits into train and test set.
    2. train: This step creates and trains a SVD model with the train set.
    3. evaluate: This step evaluate the trained model's rmse.

    Args:
        load_data: This step load a built in dataset from the Surprise library and splits into train and test set.
        train: This step creates and trains a SVD model with the train set.
        evaluate: This step evaluate the trained model's rmse.
    """
    trainset, testset = load_data()
    model = train(trainset)
    score = evaluate(model, testset)
