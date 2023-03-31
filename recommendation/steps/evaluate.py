"""A step to evaluate the train model Root Mean Squared Error (rmse)."""
import mlflow
from zenml.steps import step

from surprise import accuracy
from surprise import SVD

@step(experiment_tracker="mlflow_experiment_tracker")
def evaluate(model: SVD, testset: list) -> float:
    """Make predictions with the testset and compute for the rmse.

    Args:
        model (SVD): the trained model
        testset (list): the test dataset

    Returns:
        float: rmse
    """
    predictions = model.test(testset)
    
    rmse = accuracy.rmse(predictions)
    
    if mlflow.active_run():
        mlflow.log_metric("rmse", rmse)

    return rmse
