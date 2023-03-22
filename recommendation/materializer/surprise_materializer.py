"""Custom materializer for Surprise data and model."""
import os
import pickle

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from typing import Any, Type, Union

from surprise import AlgoBase
from surprise.trainset import Trainset

DEFAULT_FILENAME = "surprise_output.pickle"

class SurpriseMaterializer(BaseMaterializer):
    """Custom materializer for handling object created using Surprise such as dataset and models."""

    ASSOCIATED_TYPES = (AlgoBase, Trainset, list)

    def __init__(self, uri: str):
        """Initializes a materializer with the given URI."""
        self.uri = uri

    def load(self, data_type: Type[Any]) -> Union[AlgoBase, Trainset]:
        """This function loads the input from the artifact store and returns it.

        Args:
            data_type (Type[Any]): The type of the artifact.

        Returns:
            Union[AlgoBase, Trainset]: the output of the artifact.
        """
        # read from self.uri
        super().load(data_type)
        
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def save(self, obj: Any) -> None:
        """This function saves the artifact to the artifact store.

        Args:
            obj (any): The input artifact to be saved
        """
        # write `data` to self.uri
        super().save(obj)
        
        model_filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(model_filepath, "wb") as fid:
            pickle.dump(obj, fid)