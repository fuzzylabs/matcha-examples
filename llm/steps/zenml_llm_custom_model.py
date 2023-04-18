from typing import Any, Dict, List, Union, Optional
import subprocess
import numpy as np
import click

from zenml.logger import get_logger
from zenml.utils import source_utils

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_NAME = "models"
DEFAULT_LOCAL_MODEL_DIR = "/mnt/models"

logger = get_logger(__name__)
Array_Like = Union[np.ndarray, List[Any], str, bytes, Dict[str, Any]]


class ZenMLCustomLLMModel:
    """Custom model class for ZenML and Seldon for LLM.

    This class is used to implement a custom model for the Seldon Core integration,
    which is used as the main entry point for custom code execution.

    Attributes:
        name: The name of the model.
        model_uri: The URI of the model.
        predict_func: The predict function of the model.
    """

    def __init__(
        self,
        model_name: str,
        model_uri: str,
        predict_func: str,
    ):
        """Initializes a ZenMLCustomModel object.

        Args:
            model_name: The name of the model.
            model_uri: The URI of the model.
            predict_func: The predict function of the model.
        """
        self.name = model_name
        self.model_uri = model_uri
        self.predict_func = source_utils.load(predict_func)
        self.model = None
        self.tokenizer = None
        self.ready = False

    def load(self) -> bool:
        """Load the model.

        This function loads the model into memory and sets the ready flag to True.
        The model is loaded using the materializer, by saving the information of
        the artifact to a file at the preparing time and loading it again at the
        prediction time by the materializer.

        Returns:
            True if the model was loaded successfully, False otherwise.

        """
        try:

            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_uri)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_uri)
        except Exception as e:
            logger.error("Failed to load model: {}".format(e))
            return False
        self.ready = True
        return self.ready

    def predict(
        self,
        X: Array_Like,
        features_names: Optional[List[str]],
        **kwargs: Any,
    ) -> Array_Like:
        """Predict the given request.

        The main predict function of the model. This function is called by the
        Seldon Core server when a request is received. Then inside this function,
        the user-defined predict function is called.

        Args:
            X: The request to predict in a dictionary.
            features_names: The names of the features.
            **kwargs: Additional arguments.

        Returns:
            The prediction dictionary.

        Raises:
            Exception: If function could not be called.
            NotImplementedError: If the model is not ready.
            TypeError: If the request is not a dictionary.
        """
        if self.predict_func is not None:
            try:
                prediction = {"predictions": self.predict_func(self.model, self.tokenizer, X)}
            except Exception as e:
                raise Exception("Failed to predict: {}".format(e))
            if isinstance(prediction, dict):
                return prediction
            else:
                raise TypeError(
                    f"Prediction is not a dictionary. Expected dict type but got {type(prediction)}"
                )
        else:
            raise NotImplementedError("Predict function is not implemented")


@click.command()
@click.option(
    "--model_uri",
    default=DEFAULT_LOCAL_MODEL_DIR,
    type=click.STRING,
    help="The directory where the model is stored locally.",
)
@click.option(
    "--model_name",
    default=DEFAULT_MODEL_NAME,
    required=True,
    type=click.STRING,
    help="The name of the model to deploy.",
)
@click.option(
    "--predict_func",
    required=True,
    type=click.STRING,
    help="The path to the custom predict function defined by the user.",
)
def main(model_name: str, model_uri: str, predict_func: str) -> None:
    """Main function for the custom model.

    Within the deployment process, the built-in custom deployment step is used to
    to prepare the Seldon Core deployment with an entry point that calls this script,
    which then starts a subprocess to start the Seldon server and waits for requests.

    The following is an example of the entry point:
    ```
    entrypoint_command = [
        "python",
        "-m",
        "zenml.integrations.seldon.custom_deployer.zenml_custom_model",
        "--model_name",
        config.service_config.model_name,
        "--predict_func",
        config.custom_deploy_parameters.predict_function,
    ]
    ```

    Args:
        model_name: The name of the model.
        model_uri: The URI of the model.
        predict_func: The path to the predict function.
    """
    command = [
        "seldon-core-microservice",
        "steps.zenml_llm_custom_model.ZenMLCustomModel",
        "--service-type",
        "MODEL",
        "--parameters",
        (
            f'[{{"name":"model_uri","value":"{model_uri}","type":"STRING"}},'
            f'{{"name":"model_name","value":"{model_name}","type":"STRING"}},'
            f'{{"name":"predict_func","value":"{predict_func}","type":"STRING"}}]'
        ),
    ]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as ProcessError:
        logger.error(
            f"Failed to start the seldon-core-microservice process. {ProcessError}"
        )
        return


if __name__ == "__main__":
    main()
