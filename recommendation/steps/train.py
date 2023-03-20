from zenml.steps import step, Output

from surprise import SVD
from surprise.trainset import Trainset

@step
def train(trainset: Trainset) -> Output(model=SVD):
    model = SVD()

    model.fit(trainset)

    return model