"""Deploy LLM model server using seldon."""
import os
from typing import Any, Dict, cast

from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.integrations.seldon.services import SeldonDeploymentConfig
from zenml.integrations.seldon.seldon_client import SeldonDeploymentPredictorParameter
from zenml.logger import get_logger
from zenml.steps import BaseParameters, StepContext, step
from zenml.environment import Environment
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.io import fileio
from zenml.steps import STEP_ENVIRONMENT_NAME, StepEnvironment
from zenml.utils import io_utils

logger = get_logger(__name__)
DEFAULT_PT_MODEL_DIR = "hf_pt_model"
DEFAULT_TOKENIZER_DIR = "hf_tokenizer"


class LLMDeploymentParameters(BaseParameters):
    """Model Deployment Parameters."""

    # Model display name
    model_name: str

    # Timeout for model deployment task
    timeout: int

    # K8s resources configuration for Seldon Core engine
    engineResources: Dict[str, Any]

    # K8s resources configuration for predictor container
    containerResources: Dict[str, Any]


def copy_artifact(uri: str, filename: str, context: StepContext) -> str:
    """Copy an artifact to the output location of the current step.

    Args:
        uri: URI of the artifact to copy
        filename: filename for the output artifact
        context: ZenML step context

    Returns:
        str: URI of the output location

    Raises:
        RuntimeError: if the artifact is not found

    """
    served_artifact_uri = os.path.join(context.get_output_artifact_uri(), "seldon")
    fileio.makedirs(served_artifact_uri)
    if not fileio.exists(uri):
        raise RuntimeError(f"Expected artifact was not found at " f"{uri}")
    if io_utils.isdir(uri):
        io_utils.copy_dir(uri, os.path.join(served_artifact_uri, filename))
    else:
        fileio.copy(uri, os.path.join(served_artifact_uri, filename))

    return served_artifact_uri


def get_config(
    pipeline_name: str,
    run_name: str,
    step_name: str,
    model_name: str,
    artifacts_for_server: Dict[str, str],
    engine_resources: Dict[str, Any],
    container_resources: Dict[str, Any],
    kubernetes_secret_name: str,
    docker_image: str,
) -> SeldonDeploymentConfig:
    """Creates a config for Seldon Deployment.

    Args:
        pipeline_name: current pipeline name
        run_name: current run ID
        step_name: current step name
        model_name: name of the deployed model
        artifacts_for_server: a dictionary containing the name and URI of the server artifacts for server initialisation
        engine_resources: Kubernetes resources for Seldon Core engine
        container_resources: Kubernetes resources for model container
        kubernetes_secret_name: Kubernetes secret name
        docker_image: Docker image tag

    Returns:
        SeldonDeploymentConfig: configuration of the deployment
    """
    return SeldonDeploymentConfig(
        pipeline_name=pipeline_name,
        pipeline_run_id=run_name,
        pipeline_step_name=step_name,
        model_name=model_name,
        implementation="custom",
        replicas=1,
        parameters=[
            SeldonDeploymentPredictorParameter(name=name, type="STRING", value=value)
            for name, value in artifacts_for_server.items()
        ],
        resources=engine_resources,
        is_custom_deployment=True,
        secret_name=kubernetes_secret_name,
        spec={
            "containers": [
                {
                    "image": docker_image,
                    "imagePullPolicy": "Always",
                    "name": "classifier",
                    "resources": container_resources,
                }
            ]
        },
    )


@step
def deploy_llm_model(
    params: LLMDeploymentParameters,
    model_uri: str,
    tokenizer_uri: str,
    docker_image: str,
    context: StepContext,
) -> SeldonDeploymentService:
    """A step to deploy a model to Seldon Core with a custom docker image.

    Args:
        params (RecommendationDeploymentParameters): deployment parameters
        model_uri (str): Location of model artifact
        tokenizer_uri (str): Location of tokenizer artifact
        docker_image (str): image tag to use for deployment
        context (StepContext): ZenML step context

    Returns:
        SeldonDeploymentService: Seldon Core deployment service for the deployed model

    """
    # Download artifacts : model and tokenizer
    logger.info("Copying artifacts")
    model_path = os.path.join(model_uri, DEFAULT_PT_MODEL_DIR)
    served_model_uri = copy_artifact(model_path, DEFAULT_PT_MODEL_DIR, context)

    tokenizer_path = os.path.join(tokenizer_uri, DEFAULT_TOKENIZER_DIR)
    served_tokenizer_uri = copy_artifact(tokenizer_path, DEFAULT_TOKENIZER_DIR, context)

    artifacts_for_server = {
        "model_uri": served_model_uri,
        "tokenizer_uri": served_tokenizer_uri
    }

    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    # get pipeline name, step name and run id
    step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
    pipeline_name, run_name, step_name = step_env.pipeline_name, step_env.run_name, step_env.step_name

    service_config = get_config(
        pipeline_name,
        run_name,
        step_name,
        params.model_name,
        artifacts_for_server,
        params.engineResources,
        params.containerResources,
        model_deployer.kubernetes_secret_name,
        docker_image,
    )

    # deploy the service
    service = cast(
        SeldonDeploymentService,
        model_deployer.deploy_model(service_config, replace=True, timeout=params.timeout),
    )

    logger.info(
        f"Seldon Core deployment service started and reachable at:\n"
        f"    {service.prediction_url}\n"
    )

    return service
