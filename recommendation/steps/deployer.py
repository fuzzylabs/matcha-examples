from zenml.integrations.seldon.seldon_client import SeldonResourceRequirements
from zenml.integrations.seldon.services.seldon_deployment import (
    SeldonDeploymentConfig,
)
from zenml.integrations.seldon.steps.seldon_deployer import (
    CustomDeployParameters,
    SeldonDeployerStepParameters,
    seldon_custom_model_deployer_step,
)

seldon_pytorch_custom_deployment = seldon_custom_model_deployer_step(
    params=SeldonDeployerStepParameters(
        service_config=SeldonDeploymentConfig(
            model_name="seldon-svd-custom-model",
            replicas=1,
            implementation="custom",
            resources=SeldonResourceRequirements(
                limits={"cpu": "200m", "memory": "250Mi"}
            ),
        ),
        timeout=240,
        custom_deploy_parameters=CustomDeployParameters(
            predict_function="steps.svd_custom_deploy.custom_predict"
        ),
    )
)

from zenml.artifacts import ModelArtifact
from zenml.environment import Environment
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services.seldon_deployment import (
  SeldonDeploymentConfig,
  SeldonDeploymentService,
)
from zenml.steps import (
  STEP_ENVIRONMENT_NAME,
  StepContext,
  step,
)

@step()
def seldon_model_deployer_step(
  context: StepContext,
  model: ModelArtifact,
) -> SeldonDeploymentService:
  model_deployer = SeldonModelDeployer.get_active_model_deployer()

  # get pipeline name, step name and run id
  step_env = Environment()[STEP_ENVIRONMENT_NAME]

  service_config=SeldonDeploymentConfig(
      model_uri=model.uri,
      model_name="my-model",
      replicas=1,
      implementation="custom",
      pipeline_name = step_env.pipeline_name,
      pipeline_run_id = step_env.pipeline_run_id,
      pipeline_step_name = step_env.step_name,
      is_custom_deployment = True,
  )

  service = model_deployer.deploy_model(
      service_config, replace=True, timeout=300
  )

  print(
      f"Seldon deployment service started and reachable at:\n"
      f"    {service.prediction_url}\n"
  )

  return service