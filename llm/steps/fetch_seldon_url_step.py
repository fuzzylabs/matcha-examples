"""Step to get seldon url and write it to a file."""
import os
from zenml.steps import step, BaseParameters
from zenml.integrations.seldon.model_deployers.seldon_model_deployer import (
    SeldonModelDeployer,
)


class SeldonURLParameters(BaseParameters):
    """Parameters for fetching seldon endpoint."""

    # Name of pipeline
    pipeline_name: str = "llm_deployment_pipeline"

    # Name of step in the pipeline
    pipeline_step: str = "deploy_model"

    # Name of model
    model_name: str = "seldon-llm-custom-model"


@step
def get_prediction_endpoint(params: SeldonURLParameters) -> str:
    """Get the seldon endpoint for the currently deployed LLM model.

    Args:
        params (SeldonURLParameters): Parameters required for fetching seldon endpoint.

    Returns:
        str: the url endpoint.
    """
    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    deployed_services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.pipeline_step,
        model_name=params.model_name,
    )
    seldon_url = deployed_services[0].prediction_url

    secrets_file_path = "app/.streamlit/secrets.toml"

    if not os.path.exists(secrets_file_path):
        os.makedirs(os.path.basename(secrets_file_path), exist_ok=True)

    with open(secrets_file_path, "w") as f:
        f.write(f"SELDON_URL = {seldon_url}")
    return seldon_url
