from zenml.steps import step

from surprise import accuracy
from surprise import SVD

@step
def evaluate(model: SVD, testset: list) -> float:
    
    predictions = model.test(testset)

    return accuracy.rmse(predictions)
