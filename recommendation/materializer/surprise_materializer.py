import os
import pickle

from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from typing import Any, Type, Union

from surprise import AlgoBase
from surprise.trainset import Trainset

DEFAULT_FILENAME = "surprise_output.pickle"

class SurpriseMaterializer(BaseMaterializer):

    ASSOCIATED_TYPES = (AlgoBase, Trainset, list)

    def __init__(self, uri: str):
        """Initializes a materializer with the given URI."""
        self.uri = uri

    def load(self, data_type: Type[Any]) -> Union[AlgoBase, Trainset]:
        """Write logic here to load the data of an artifact.

        Args:
            data_type: What type the artifact data should be loaded as.

        Returns:
            The data of the artifact.
        """
        # read from self.uri
        super().load(data_type)
        
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            obj = pickle.load(fid)
        return obj

    def save(self, obj: Any) -> None:
        """Write logic here to save the data of an artifact.

        Args:
            data: The data of the artifact to save.
        """
        # write `data` to self.uri
        super().save(obj)
        
        model_filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(model_filepath, "wb") as fid:
            pickle.dump(obj, fid)