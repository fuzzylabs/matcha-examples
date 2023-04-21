"""Deployment pipeline for the LLM example."""
from zenml.pipelines import pipeline


@pipeline
def llm_deployment_pipeline(fetch_trained_model, build_docker_image, deploy_llm_model):
    """Pipeline to deploy fine-tuned LLM model.

    Args:
        fetch_trained_model : A step to fetch path to trained model and decision to deploy the model. 
        build_docker_image: This step build a custom Docker image for deployment
        deploy_llm_model: A step to deploy model using Seldon.
    """
    build_docker_image.after(fetch_trained_model)

    # Fetch the trained model path, tokenizer path and decision
    model_uri, tokenizer_uri, decision = fetch_trained_model()

    # Build custom docker image
    image_tag = build_docker_image()

    # Deploy the model
    deploy_llm_model(model_uri, tokenizer_uri, image_tag)
