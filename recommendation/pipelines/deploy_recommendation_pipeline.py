"""A pipeline to deploy recommendation model."""
from zenml.pipelines import pipeline
from zenml.config import DockerSettings

# environment related issue: https://github.com/NicolasHug/Surprise/issues/364
docker_settings = DockerSettings(apt_packages=["gcc", "build-essential"], 
                                 environment={"SURPRISE_DATA_FOLDER": "/tmp"})


@pipeline(settings={"docker": docker_settings})
def recommendation_deployment_pipeline(
    fetch_model, deployment_trigger, deploy_model
):
    """Recommendation deployment pipeline.
    Args:
        fetch_model: This step fetches a model artifact from a training pipeline run
        deploy_model: This step deploys the model with Seldon Core
    """
    model = fetch_model()
    
    decision = deployment_trigger(model)
    
    deploy_model(decision, model)
