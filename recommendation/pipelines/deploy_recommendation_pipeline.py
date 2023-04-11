"""A pipeline to deploy recommendation model."""
from zenml.pipelines import pipeline
from zenml.config import DockerSettings

docker_settings = DockerSettings(apt_packages=["gcc", "build-essential"])


@pipeline(settings={"docker": docker_settings})
def recommendation_deployment_pipeline(
    fetch_model, deployment_trigger, deploy_model
):
    """Recommendation deployment pipeline.
    Args:
        fetch_model: This step fetches a model artifact from a training pipeline run
        deploy_model: This step deploys the model with Seldon Core
    """
    # Fetch model from nft_embedding_pipeline
    model = fetch_model()
    
    decision = deployment_trigger(model)
    
    # Deploy model with Seldon Core
    deploy_model(decision, model)
