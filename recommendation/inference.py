"""A inference script to query the deployed recommendation model."""
import requests
import json 
import click

from zenml.integrations.seldon.model_deployers.seldon_model_deployer import SeldonModelDeployer

PIPELINE_NAME = 'recommendation_deployment_pipeline'
PIPELINE_STEP = 'deploy_model'
MODEL_NAME = 'seldon-svd-custom-model'


def _get_prediction_endpoint() -> str:
    """Get the endpoint for the currently deployed recommendation model.

    Returns:
        str: the url endpoint.
    """
    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    deployed_services = model_deployer.find_model_server(
        pipeline_name=PIPELINE_NAME,
        pipeline_step_name=PIPELINE_STEP,
        model_name=MODEL_NAME
    )

    return deployed_services[0].prediction_url


def _create_payload(user: str, movie: str) -> dict:
    """Create a payload from the user input to send to the recommendation model.

    Args:
        user (str): the id of the user.
        movie (str): the id of the movie.

    Returns:
        dict: the payload to send in the correct format.
    """
    return {
        "data": {
            "ndarray": [{"uid": str(user), "iid": str(movie)}]
        }
    }


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
        headers={'Content-Type': 'application/json'}
    )

    return json.loads(response.text)['jsonData']['predictions']


def _output(predictions: dict):
    """Output the results to the terminal.

    Args:
        predictions (dict): the predictions produced by the deployed model.
    """
    for pred in predictions:
        rating = round(pred['est'], 2)
        print(f"User {pred['uid']} is predicted to give the movie ({pred['iid']}) a rating of: {rating} out of 5.")
    

@click.command()
@click.option('--user', default=1, help='the user id.', type=click.IntRange(1, 943))
@click.option('--movie', default=1, help='the movie id.', type=click.IntRange(1, 1682))
def main(user: int, movie: int):
    """The main runner function.

    Args:
        user (int): the user inputted user id.
        movie (int): the user inputted movie id.
    """
    endpoint = _get_prediction_endpoint()
    payload = _create_payload(user, movie)
    predictions = _get_predictions(endpoint, payload)
    _output(predictions)


if __name__ == '__main__':
    main()
