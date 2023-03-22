"""A step to train a Singular value decomposition (SVD) model."""
from zenml.steps import step, Output

from surprise import SVD
from surprise.trainset import Trainset


@step
def train(trainset: Trainset) -> Output(model=SVD):
    """Train and return a SVD model.

    Args:
        trainset (Trainset): the data for model training.

    Returns:
        SVD: the trained SVD model
    """
    model = SVD()

    model.fit(trainset)

    return model