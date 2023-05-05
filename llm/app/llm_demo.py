import streamlit as st
import json
from zenml.integrations.seldon.model_deployers.seldon_model_deployer import (
    SeldonModelDeployer,
)

st.title("LLM Summarization Demo")

PIPELINE_NAME = "llm_deployment_pipeline"
PIPELINE_STEP = "deploy_model"
MODEL_NAME = "seldon-llm-custom-model"


def _get_prediction_endpoint() -> str:
    """Get the endpoint for the currently deployed LLM model.

    Returns:
        str: the url endpoint.
    """
    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    deployed_services = model_deployer.find_model_server(
        pipeline_name=PIPELINE_NAME,
        pipeline_step_name=PIPELINE_STEP,
        model_name=MODEL_NAME,
    )

    return deployed_services[0].prediction_url


def fetch_summary(txt: str) -> str:
    """Query seldon endpoint to fetch the summary.

    Args:
        txt (str): Input text to summarize

    Returns:
        str: Summarized text
    """
    seldon_url = _get_prediction_endpoint()
    if seldon_url is None:
        st.write("Hmm, seldon endpoint is not provisioned yet!")
    else:
        pass


def read_examples(file_path: str = "example.json") -> dict:
    """Read sample examples for LLM summarization demo.

    Args:
        file_path (str, optional): Path to json file. Defaults to 'example.json'.

    Returns:
        dict: Dictionary containing examples.
    """
    with open(file_path, "r") as myfile:
        data = myfile.read()
    return json.loads(data)


def switch_examples(data: dict) -> str:
    """Switch between different examples.

    Args:
        data (dict): Dictionary containing examples.

    Returns:
        str: Input text to summarize.
    """
    pages = ["Example 1", "Example 2", "Example 3"]
    page = st.radio("Test Examples", pages)

    if page == "Example 1":
        text = data["example1"]

    if page == "Example 2":
        text = data["example2"]

    if page == "Example 3":
        text = data["example3"]

    input_text = st.text_area(label="Text to summarize", value=text, height=400)
    return input_text


def main():
    data = read_examples(file_path="example.json")

    txt = switch_examples(data)

    result = st.button(label="Ready")

    if result:
        summarized_text = st.text_area(
            label="Summarized Text",
            value=fetch_summary(txt),
            height=200,
        )


if __name__ == "__main__":
    main()
