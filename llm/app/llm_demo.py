"""Streamlit application."""
import os
import requests
import streamlit as st
import json
from zenml.integrations.seldon.model_deployers.seldon_model_deployer import (
    SeldonModelDeployer,
)

st.title("LLM Summarization Demo")

PIPELINE_NAME = "llm_deployment_pipeline"
PIPELINE_STEP = "deploy_model"
MODEL_NAME = "seldon-llm-custom-model"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
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


def _create_payload(input_text: str) -> dict:
    """Create a payload from the user input to send to the LLM model.

    Args:
        input_text (str): Input text to summarize.

    Returns:
        dict: the payload to send in the correct format.
    """
    return {"data": {"ndarray": [{"text": str(input_text)}]}}


def _get_predictions(prediction_endpoint: str, payload: dict) -> dict:
    """Using the prediction endpont and payload, make a prediction request to the deployed model.

    Args:
        prediction_endpoint (str): the url endpoint.
        payload (dict): the payload to send to the model.

    Returns:
        dict: the predictions from the model.
    """
    response = requests.post(
        url=prediction_endpoint,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(response.text)["jsonData"]["predictions"][0]


def fetch_summary(seldon_url: str, txt: str) -> str:
    """Query seldon endpoint to fetch the summary.

    Args:
        seldon_url (str): Seldon endpoint
        txt (str): Input text to summarize

    Returns:
        str: Summarized text
    """
    with st.spinner("Applying LLM Magic..."):
        payload = _create_payload(txt)
        summary_txt = _get_predictions(seldon_url, payload)
    return summary_txt


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
    example_file_path = os.path.join(BASE_DIR, "example.json")
    data = read_examples(file_path=example_file_path)

    txt = switch_examples(data)

    result = st.button(label="Ready")

    if result:
        seldon_url = _get_prediction_endpoint()

        if seldon_url is None:
            st.write("Hmm, seldon endpoint is not provisioned yet!")

        else:
            summarized_text = st.text_area(
                label="Summarized Text",
                value=fetch_summary(seldon_url, txt),
                height=200,
            )


if __name__ == "__main__":
    main()
