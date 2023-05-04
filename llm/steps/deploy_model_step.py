# """Deploy LLM model server using seldon."""
from zenml.integrations.seldon.seldon_client import SeldonResourceRequirements
from zenml.integrations.seldon.services.seldon_deployment import (
    SeldonDeploymentConfig,
)
from zenml.integrations.seldon.steps.seldon_deployer import (
    CustomDeployParameters,
    SeldonDeployerStepParameters,
)
from steps.seldon_llm_custom_deployer_step import seldon_llm_model_deployer_step

seldon_llm_custom_deployment = seldon_llm_model_deployer_step(
    params=SeldonDeployerStepParameters(
        service_config=SeldonDeploymentConfig(
            model_name="seldon-llm-custom-model",
            replicas=1,
            implementation="custom",
            resources=SeldonResourceRequirements(
                limits={"cpu": "500m", "memory": "900Mi"}
            ),
        ),
        timeout=300,
        custom_deploy_parameters=CustomDeployParameters(
            predict_function="steps.llm_custom_predict.custom_predict"
        ),
    )
)
