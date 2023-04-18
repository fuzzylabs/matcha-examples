"""Server for LLM summarization fine-tuned model."""
import logging
from typing import Optional, Any
import os
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class LLMServer(object):

    def __init__(
        self,
        model_uri: str = None,
        tokenizer_uri: str = None,
        working_directory: Optional[str] = None,
    ):
        """Initialise model server.

        Args:
            hf_dataset_uri: URI pointing to the huggingface dataset
            faiss_index_uri: URI pointing to the FAISS index
            working_directory (Optional[str]): directory to download artifacts to, defaults to the directory of the script
        """
        super().__init__()

        if working_directory is None:
            working_directory = os.path.dirname(os.path.abspath(__file__))

        # TODO: Copy artifacts to working directory

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.trained_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.trained_model_path)

    def preprocess_input(self, input_text: str) -> np.ndarray:
        """Preprocess input text using tokenizer.

        Args:
            input_text (str): input text

        Returns:
        """
        input_id = self.tokenizer(input_text, return_tensors="pt").input_ids
        return input_id

    def predict(self, X: np.ndarray, feature_names: np.ndarray, **kwargs: Any):
        """Predict function in Seldon format to get predictions.

        Args:
            X (np.ndarray): input data
            feature_names (np.ndarray): names of the features
            **kwargs: extra arguments

        Returns:.
        """
        try:
            input_id = self.preprocess_input(X["text"])
            outputs = self.model.generate(input_id,
                                          max_new_tokens=100,
                                          num_beams=5,
                                          early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.exception("Exception during predict", e)
