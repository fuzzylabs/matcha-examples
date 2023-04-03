"""A pipeline to deploy recommendation model."""
from zenml.pipelines import pipeline

@pipeline
def recommendation_deployment_pipeline(
    fetch_model, deploy_model
):
    """Recommendation deployment pipeline.
    Args:
        fetch_model: This step fetches a model artifact from a training pipeline run
        deploy_model: This step deploys the model with Seldon Core
    """
    # Fetch model from nft_embedding_pipeline
    model = fetch_model()
    
    # Deploy model with Seldon Core
    deploy_model(model)
