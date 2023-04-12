"""Custom Seldon deployer step."""
from zenml.integrations.seldon.seldon_client import SeldonResourceRequirements
from zenml.integrations.seldon.services.seldon_deployment import (
    SeldonDeploymentConfig,
)
from zenml.integrations.seldon.steps.seldon_deployer import (
    CustomDeployParameters,
    SeldonDeployerStepParameters,
    seldon_custom_model_deployer_step,
)

seldon_surprise_custom_deployment = seldon_custom_model_deployer_step(
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