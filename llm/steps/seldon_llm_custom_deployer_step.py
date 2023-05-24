# Derived from zenml seldon integration; source : https://github.com/zenml-io/zenml/blob/main/src/zenml/integrations/seldon/steps/seldon_deployer.py
"""Custom zenml deployer step for Seldon LLM."""
import os
from typing import cast
from zenml.environment import Environment
from zenml.exceptions import DoesNotExistException

from zenml.integrations.seldon.constants import (
    SELDON_CUSTOM_DEPLOYMENT,
    SELDON_DOCKER_IMAGE_KEY,
)
from zenml.integrations.seldon.model_deployers.seldon_model_deployer import (
    SeldonModelDeployer,
)
from zenml.integrations.seldon.seldon_client import (
    create_seldon_core_custom_spec,
)
from zenml.integrations.seldon.steps.seldon_deployer import SeldonDeployerStepParameters
from zenml.integrations.seldon.services.seldon_deployment import (
    SeldonDeploymentService,
)

from zenml.io import fileio
from zenml.logger import get_logger

from zenml.steps import (
    STEP_ENVIRONMENT_NAME,
    StepEnvironment,
    step,
)
from zenml.steps.step_context import StepContext
from zenml.utils import io_utils


logger = get_logger(__name__)
DEFAULT_PT_MODEL_DIR = "hf_pt_model"
DEFAULT_TOKENIZER_DIR = "hf_tokenizer"


def copy_artifact(uri: str, filename: str, context: StepContext) -> str:
    """Copy an artifact to the output location of the current step.

    Args:
        uri (str): URI of the artifact to copy
        filename (str): filename for the output artifact
        context (StepContext): ZenML step context

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


@step(enable_cache=False, extra={SELDON_CUSTOM_DEPLOYMENT: True})
def seldon_llm_model_deployer_step(
    deploy_decision: bool,
    model_uri: str,
    tokenizer_uri: str,
    params: SeldonDeployerStepParameters,
    context: StepContext,
) -> SeldonDeploymentService:
    """Seldon Core custom model deployer pipeline step for LLM models.
    This step can be used in a pipeline to implement the
    the process required to deploy a custom model with Seldon Core.

    Args:
        deploy_decision (bool): whether to deploy the model or not
        model_uri (str): The URI of huggingface model
        tokenizer_uri (str): The URI of huggingface tokenizer
        params (SeldonDeployerStepParameters): parameters for the deployer step
        context (StepContext): the step context

    Raises:
        ValueError: if the custom deployer is not defined
        DoesNotExistException: if an entity does not exist raise an exception

    Returns:
        SeldonDeploymentService: Seldon Core deployment service
    """
    # verify that a custom deployer is defined
    if not params.custom_deploy_parameters:
        raise ValueError(
            "Custom deploy parameter is required as part of the step configuration this parameter is",
            "the path of the custom predict function",
        )
    # get the active model deployer
    model_deployer = cast(
        SeldonModelDeployer, SeldonModelDeployer.get_active_model_deployer()
    )

    # get pipeline name, step name, run id
    step_env = cast(StepEnvironment, Environment()[STEP_ENVIRONMENT_NAME])
    pipeline_name = step_env.pipeline_name
    run_name = step_env.run_name
    step_name = step_env.step_name

    # update the step configuration with the real pipeline runtime information
    params.service_config.pipeline_name = pipeline_name
    params.service_config.pipeline_run_id = run_name
    params.service_config.pipeline_step_name = step_name
    params.service_config.is_custom_deployment = True

    # fetch existing services with the same pipeline name, step name and
    # model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=params.service_config.model_name,
    )
    # even when the deploy decision is negative if an existing model server
    # is not running for this pipeline/step, we still have to serve the
    # current model, to ensure that a model server is available at all times
    if not deploy_decision and existing_services:
        logger.info(
            f"Skipping model deployment because the model quality does not"
            f" meet the criteria. Reusing the last model server deployed by step "
            f"'{step_name}' and pipeline '{pipeline_name}' for model "
            f"'{params.service_config.model_name}'..."
        )
        service = cast(SeldonDeploymentService, existing_services[0])
        # even when the deployment decision is negative, we still need to start
        # the previous model server if it is no longer running, to ensure that
        # a model server is available at all times
        if not service.is_running:
            service.start(timeout=params.timeout)
        return service

    # entrypoint for starting Seldon microservice deployment for custom model
    entrypoint_command = [
        "python",
        "-m",
        "steps.zenml_llm_custom_model",
        "--model_name",
        params.service_config.model_name,
        "--predict_func",
        params.custom_deploy_parameters.predict_function,
    ]

    # verify if there is an active stack before starting the service
    if not context.stack:
        raise DoesNotExistException(
            "No active stack is available. "
            "Please make sure that you have registered and set a stack."
        )

    image_name = step_env.step_run_info.get_image(key=SELDON_DOCKER_IMAGE_KEY)

    # Copy artifacts
    model_path = os.path.join(model_uri, DEFAULT_PT_MODEL_DIR)
    served_model_uri = copy_artifact(model_path, DEFAULT_PT_MODEL_DIR, context)

    tokenizer_path = os.path.join(tokenizer_uri, DEFAULT_TOKENIZER_DIR)
    copy_artifact(tokenizer_path, DEFAULT_TOKENIZER_DIR, context)

    # prepare the service configuration for the deployment
    service_config = params.service_config.copy()
    service_config.model_uri = served_model_uri

    # create the specification for the custom deployment
    service_config.spec = create_seldon_core_custom_spec(
        model_uri=service_config.model_uri,
        custom_docker_image=image_name,
        secret_name=model_deployer.kubernetes_secret_name,
        command=entrypoint_command,
    )

    # deploy the service
    service = cast(
        SeldonDeploymentService,
        model_deployer.deploy_model(
            service_config, replace=True, timeout=params.timeout
        ),
    )

    logger.info(
        f"Seldon Core deployment service started and reachable at:\n"
        f"    {service.prediction_url}\n"
    )

    return service
