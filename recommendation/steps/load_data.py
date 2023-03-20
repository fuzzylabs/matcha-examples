from zenml.steps import step, Output, BaseParameters

from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise.trainset import Trainset


class DataParameters(BaseParameters):
    test_size = 0.25

@step
def load_data(params: DataParameters) -> Output(trainset=Trainset, testset=list):
    data = Dataset.load_builtin("ml-100k")

    trainset, testset = train_test_split(data, test_size=params.test_size)

    return trainset, testset