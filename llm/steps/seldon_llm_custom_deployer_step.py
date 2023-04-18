"""Custom zenml deployer step for Seldon LLM."""
import os
from typing import cast
from zenml.environment import Environment
from zenml.exceptions import DoesNotExistException
from zenml.constants import MODEL_METADATA_YAML_FILE_NAME
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
from zenml.materializers import UnmaterializedArtifact
from zenml.steps import (
    STEP_ENVIRONMENT_NAME,
    StepEnvironment,
    step,
)
from zenml.steps.step_context import StepContext
from zenml.utils import io_utils
from zenml.utils.materializer_utils import save_model_metadata

logger = get_logger(__name__)


@step(enable_cache=False, extra={SELDON_CUSTOM_DEPLOYMENT: True})
def seldon_llm_model_deployer_step(
    deploy_decision: bool,
    params: SeldonDeployerStepParameters,
    context: StepContext,
    model: UnmaterializedArtifact,
) -> SeldonDeploymentService:
    """Seldon Core custom model deployer pipeline step for LLM models.

    This step can be used in a pipeline to implement the
    the process required to deploy a custom model with Seldon Core.

    Args:
        deploy_decision: whether to deploy the model or not
        params: parameters for the deployer step
        model: the model artifact to deploy
        context: the step context

    Raises:
        ValueError: if the custom deployer is not defined
        DoesNotExistException: if an entity does not exist raise an exception

    Returns:
        Seldon Core deployment service
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
    params.service_config.run_name = run_name
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

    # copy the model files to new specific directory for the deployment
    served_model_uri = os.path.join(
        context.get_output_artifact_uri(), "seldon"
    )
    fileio.makedirs(served_model_uri)
    io_utils.copy_dir(model.uri, served_model_uri)

    # save the model artifact metadata to the YAML file and copy it to the
    # deployment directory
    model_metadata_file = save_model_metadata(model)
    fileio.copy(
        model_metadata_file,
        os.path.join(served_model_uri, MODEL_METADATA_YAML_FILE_NAME),
    )

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
