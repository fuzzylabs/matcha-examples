"""Deployment pipeline for the LLM example."""
from zenml.pipelines import pipeline


@pipeline
def llm_deployment_pipeline(fetch_trained_model, deploy_model, get_prediction_endpoint):
    """Pipeline to deploy fine-tuned LLM model.

    Args:
        fetch_trained_model : A step to fetch path to trained model and decision to deploy the model.
        deploy_model: A step to deploy LLM model using Seldon.
        get_prediction_endpoint: A step to fetch seldon endpoint URL.
    """
    get_prediction_endpoint.after(deploy_model)

    # Fetch the trained model path, tokenizer path and decision
    model_uri, tokenizer_uri, decision = fetch_trained_model()

    # Deploy the model
    deploy_model(decision, model_uri, tokenizer_uri)

    get_prediction_endpoint()
