import requests
import json 
from zenml.integrations.seldon.model_deployers.seldon_model_deployer import (
    SeldonModelDeployer,
)
from zenml.integrations.seldon.services.seldon_deployment import (
    SeldonDeploymentService,
)


def get_user_input() -> dict:
    uid = input("What user ID would you like to use?: ")
    iid = input("What item ID would you like to predict the rating for?: ")
    return {"data": {"ndarray": [{"uid": uid, "iid": iid}]}}


def post_prediction_request(prediction_url: str, data: dict) -> dict:
    headers = {'Content-Type': 'application/json'}
    response = requests.post(prediction_url, headers=headers, data=json.dumps(data))
    
    return json.loads(response.text)['jsonData']


def output_prediction_response(response:str) -> None:
    for prediction in response['predictions']:
        print(f"User with ID {prediction['uid']} is predicted to give movie with ID {prediction['iid']} a rating of {round(prediction['est'], 3)} (to three d.p).")


def get_prediction_url() -> str:
    # model_deployer = SeldonModelDeployer.get_active_model_deployer()

    # services = model_deployer.find_model_server(
    #     pipeline_name='recommendation_deployment_pipeline',
    #     pipeline_step_name='deploy_model',
    #     model_name='seldon-svd-custom-model',
    # )

    # return services.prediction_url
    return 'http://REDACTED URL'


def main():
    """Send request to prediction server."""
    print("Prediction Server Request Script")
    sample_dict = get_user_input()
    prediction_url = get_prediction_url()
    response = post_prediction_request(prediction_url=prediction_url, data=sample_dict)
    output_prediction_response(response)
    

if __name__ == "__main__":
    main()